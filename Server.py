from Model import get_model
from Utils import *

# Server configuration
SERVER_HOST = config['server']['host']
SERVER_PORT = config['server']['comm_port']

UPLOAD_MODEL = config['paths']['upload_model']
UPLOAD_CONFIG = config['paths']['upload_config']

sync_mode = config['training']['sync_mode']
Run_Clients = int(config['dataset']['num_client'] * config['training']['async_concurrency'])
Aggregate_Clients = Run_Clients
if sync_mode:
    Run_Clients = int(config['dataset']['num_client'] * config['training']['sync_participant'])
else:
    Aggregate_Clients -= 1

print(f'* System Clients: {Run_Clients} & {Aggregate_Clients}')

# Client states
clients = {}
training_clients = set()
waiting_clients = set()
available_clients = set()
activating_client = set()
clients_lock = asyncio.Lock()
model_updates_lock = asyncio.Lock()

model_updates = []
model_updates_ids = []
global_model = get_model(config['dataset']['dataname'], config['dataset']['modelname'])
global_model_version = 0
stats_clients_latency = {}
stats_clients_staleness = {}
upload_ratio = config['training']['upload_ratio']
download_ratio = config['training']['download_ratio']
consistent_ratio = config['training']['consistent_ratio']

client_id_counter = 0

start_training_time = None
training_round = 0

clients_configs = {}
clients_start_time = {}

# Set up logging
logging.basicConfig(
    filename=config['paths']['log_path_server_manager'],
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

async def save_config(config_dict, filename):
    async with aiofiles.open(filename, 'w') as f:
        await f.write(json.dumps(config_dict, indent=4))

async def handle_client(reader, writer, client_id):
    global model_updates, global_model

    clients_configs[client_id] = cp.deepcopy(config['training'])
    config_path = os.path.join(UPLOAD_CONFIG, f"{client_id}.json")

    os.makedirs(UPLOAD_CONFIG, exist_ok=True)

    await save_config(clients_configs[client_id], config_path)
    stats_clients_latency[client_id] = []
    stats_clients_staleness[client_id] = []
    logging.info(f"Client connected with ID: {client_id}")

    async with clients_lock:
        clients[client_id] = (reader, writer)
        available_clients.add(client_id)

    try:
        while True:
            message = (await reader.read(1024)).decode('utf-8')
            logging.info(f"Message from {client_id}: {message}")

            if 'available' in message:
                async with clients_lock:
                    if client_id not in training_clients and client_id not in waiting_clients and client_id not in available_clients:
                        available_clients.add(client_id)

            elif 'train_ack' in message:
                async with clients_lock:
                    activating_client.add(client_id)
                    logging.info(f'{client_id} is activated for training')

            elif 'completed' in message:
                load_local = False
                local_model = None
                original_shape_dict = {key: value.shape for key, value in global_model.state_dict().items()}
                local_model_path = os.path.join(UPLOAD_MODEL, f"{client_id}.pth.gz")

                while not load_local:
                    try:
                        if os.path.exists(local_model_path):
                            async with aiofiles.open(local_model_path, 'rb') as f:
                                compressed_data = await f.read()
                                with gzip.GzipFile(fileobj=BytesIO(compressed_data), mode='rb') as decompressed_file:
                                    decompressed_data = decompressed_file.read()
                                    local_model = decompress_model(decompressed_data, original_shape_dict)
                                    load_local = True
                        else:
                            raise FileNotFoundError(f"Model file not found: {local_model_path}")
                    except Exception as e:
                        logging.error(f"Error loading local model {local_model_path}: {e}")
                        await asyncio.sleep(0.1)

                end_time = time.time()
                latency = end_time - clients_start_time[client_id]
                stats_clients_latency[client_id].append(latency)
                staleness = np.mean(stats_clients_staleness[client_id][-10:]) if stats_clients_staleness[client_id] else 0
                logging.info(f"{client_id} completed training with latency: {latency:.2f} seconds & staleness: {staleness:.2f}")
                print(f"{client_id} completed training with latency: {latency:.2f} seconds & staleness: {staleness:.2f}")

                async with model_updates_lock:
                    model_updates.append(local_model)
                    model_updates_ids.append(client_id)

                async with clients_lock:
                    training_clients.remove(client_id)
                    activating_client.remove(client_id)
                    waiting_clients.add(client_id)
                    logging.info(f"{client_id} is now waiting. Training clients: {len(training_clients)}, Waiting clients: {len(waiting_clients)}")
                    print(f"{client_id} is now waiting. Training clients: {len(training_clients)}, Waiting clients: {len(waiting_clients)}")

                writer.write("waiting".encode('utf-8'))
                await writer.drain()

    except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
        async with clients_lock:
            if client_id in clients:
                del clients[client_id]
            training_clients.discard(client_id)
            waiting_clients.discard(client_id)
            available_clients.discard(client_id)
        logging.info(
            f"{client_id} disconnected. Training clients: {len(training_clients)}, Waiting clients: {len(waiting_clients)}, Available clients: {len(available_clients)}")
        writer.close()
        await writer.wait_closed()


async def aggregate_models():
    global model_updates, model_updates_ids, global_model, global_model_version, sync_mode, training_round, start_training_time
    while True:
        await asyncio.sleep(0.1)
        async with model_updates_lock:
            if len(model_updates) >= Aggregate_Clients:
                global_model_copy = cp.deepcopy(global_model)
                model_updates_copy = model_updates[:]
                model_updates_ids_copy = model_updates_ids[:]
                model_updates.clear()
                model_updates_ids.clear()
            else:
                model_updates_copy = []
                model_updates_ids_copy = []

        if len(model_updates_copy) >= Aggregate_Clients:
            global_model_paras = global_model_copy.state_dict()
            global_model_paras_partial = compress_model_untarget(global_model_paras, download_ratio)

            if sync_mode:
                global_model_path = os.path.join(UPLOAD_MODEL, 'global_model.pth.gz')
                save_compressed_state_dict(global_model_paras_partial, global_model_path)
            else:
                for i, client_id in enumerate(model_updates_ids_copy):
                    logging.info(f"* client id: {client_id}")

                    if consistent_ratio:
                        goal_model_paras = model_updates_copy[i]
                        global_model_paras_partial = compress_model_target(global_model_paras, goal_model_paras)

                    global_model_path = os.path.join(UPLOAD_MODEL, f'global_model_{client_id}.pth.gz')
                    save_compressed_state_dict(global_model_paras_partial, global_model_path)

            logging.info("Model aggregation complete and activate waiting clients to be available")
            print("Model aggregation complete and activate waiting clients to be available")

            async with clients_lock:
                for client_id in list(waiting_clients):
                    waiting_clients.remove(client_id)
                    available_clients.add(client_id)
                    staleness = global_model_version - clients_configs[client_id]["model_version"]
                    stats_clients_staleness[client_id].append(staleness)
                    clients_configs[client_id]["model_version"] = global_model_version
                    config_path = os.path.join(UPLOAD_CONFIG, f"{client_id}.json")
                    await save_config(clients_configs[client_id], config_path)
                    reader, writer = clients[client_id]
                    writer.write("available".encode('utf-8'))
                    await writer.drain()

            if not sync_mode:
                global_model_version += 1

            round_training_time = time.time()
            training_round += 1
            logging.info(f"* Current training time: {round_training_time - start_training_time} / {training_round} round *")
            print(f"* Current training time: {round_training_time - start_training_time} / {training_round} round *")

async def send_train_signal(client_id):
    global start_training_time
    if start_training_time == None:
        start_training_time = time.time()
    logging.info(f"Sending train signal to {client_id}")
    async with clients_lock:
        if client_id in training_clients or client_id not in available_clients:
            return

        reader, writer = clients[client_id]
        training_clients.add(client_id)
        available_clients.remove(client_id)

    while True:
        try:
            writer.write("train".encode('utf-8'))
            await writer.drain()
            if client_id in activating_client:
                break
        except Exception as e:
            logging.error(f"Error sending train signal to client {client_id}: {e}, retry later")
            await asyncio.sleep(1)


async def async_activate_clients():
    global training_clients, Run_Clients
    while True:
        await asyncio.sleep(0.01)
        async with clients_lock:
            if len(training_clients) < Run_Clients and available_clients:
                selected_clients = rd.sample(list(available_clients), min(Run_Clients - len(training_clients), len(available_clients)))
                logging.info(f"Selected clients for training: {selected_clients} / {len(training_clients)}")
                print(f"Selected clients for training: {selected_clients} / {len(training_clients)}")
                for client_id in selected_clients:
                    clients_start_time[client_id] = time.time()
                    asyncio.create_task(send_train_signal(client_id))

async def sync_activate_clients():
    global available_clients, training_clients, waiting_clients, Run_Clients
    while True:
        await asyncio.sleep(0.01)
        async with clients_lock:
            if len(available_clients) >= Run_Clients and len(training_clients) == 0 and len(waiting_clients) == 0:
                selected_clients = rd.sample(list(available_clients), Run_Clients)
                logging.info(f"Selected clients for training: {selected_clients}")
                print(f"Selected clients for training: {selected_clients}")
                for client_id in selected_clients:
                    clients_start_time[client_id] = time.time()
                    asyncio.create_task(send_train_signal(client_id))

async def accept_clients(reader, writer):
    global client_id_counter
    client_id = f"client_{client_id_counter}"
    client_id_counter += 1
    writer.write(client_id.encode('utf-8'))
    await writer.drain()

    global_model_path = os.path.join(UPLOAD_MODEL, f'global_model_{client_id}.pth.gz')

    os.makedirs(UPLOAD_MODEL, exist_ok=True)

    save_compressed_state_dict(global_compressed, global_model_path)

    asyncio.create_task(handle_client(reader, writer, client_id))

async def main():
    global global_model, global_compressed

    logging.info("Saving initial model to server")
    global_model_paras = global_model.state_dict()
    global_compressed = compress_model_untarget(global_model_paras, 1)
    if sync_mode:
        global_model_path = os.path.join(UPLOAD_MODEL, 'global_model.pth.gz')
        save_compressed_state_dict(global_compressed, global_model_path)

    logging.info(f"Server started and listening on {SERVER_HOST}:{SERVER_PORT}")
    print(f"Server started and listening on {SERVER_HOST}:{SERVER_PORT}")

    server = await asyncio.start_server(accept_clients, SERVER_HOST, SERVER_PORT)

    asyncio.create_task(aggregate_models())
    if not sync_mode:
        asyncio.create_task(async_activate_clients())
    else:
        asyncio.create_task(sync_activate_clients())

    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    asyncio.run(main())
