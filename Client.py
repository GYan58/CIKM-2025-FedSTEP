from Model import get_model
from Utils import *

SERVER_HOST = config['server']['host']
SERVER_COMM_PORT = config['server']['comm_port']
SERVER_LINK_PORT = config['server']['trans_port']

UPLOAD_URL = f'http://{SERVER_HOST}:{SERVER_LINK_PORT}/upload'
DOWNLOAD_URL_CONFIG = f'http://{SERVER_HOST}:{SERVER_LINK_PORT}/download/'
DOWNLOAD_URL_DATA = f'http://{SERVER_HOST}:{SERVER_LINK_PORT}/download/'
DOWNLOAD_URL_MODEL = f'http://{SERVER_HOST}:{SERVER_LINK_PORT}/download/global_model'
NOTIFY_URL = f'http://{SERVER_HOST}:{SERVER_LINK_PORT}/download_complete'

sync_mode = config['training']['sync_mode']
dataset_cache = {}

async def download_dataset(session, client_id):
    start_time = time.time()
    dataset = None

    async with session.get(f"{DOWNLOAD_URL_DATA}{client_id}.pkl") as response:
        if response.status == 200:
            file_buffer = BytesIO()
            while chunk := await response.content.read(1024):
                file_buffer.write(chunk)

            data = {'filename': f'{client_id}.pkl', 'start_time': start_time}
            async with session.post(NOTIFY_URL, json=data) as notify_response:
                if notify_response.status == 200:
                    print("Download dataset completion successfully logged on the server.")
                else:
                    print(f"Failed to log download dataset completion on the server. "
                          f"Status code: {notify_response.status}, content: {await notify_response.text()}")

            file_buffer.seek(0)
            dataset = pickle.load(file_buffer)
        else:
            print(f"Failed to download the client dataset. Status code: {response.status}, content: {await response.text()}")

    return dataset

async def download_model_config(session, client_id):
    global sync_mode
    start_time = time.time()
    configs, global_state_dict = None, None

    url_model = f"{DOWNLOAD_URL_MODEL}_{client_id}" if not sync_mode else DOWNLOAD_URL_MODEL

    async with session.get(f"{DOWNLOAD_URL_CONFIG}{client_id}.json") as response:
        if response.status == 200:
            configs = await response.json()
        else:
            print(f"Failed to download the config file. Status code: {response.status}, content: {await response.text()}")

    async with session.get(f"{url_model}.pth.gz") as response:
        if response.status == 200:
            file_buffer = BytesIO()
            while chunk := await response.content.read(1024):
                file_buffer.write(chunk)

            data = {'filename': f'global_model_{client_id}.pth.gz', 'start_time': start_time}
            async with session.post(NOTIFY_URL, json=data) as notify_response:
                if notify_response.status == 200:
                    print("Download model completion successfully logged on the server.")
                else:
                    print(f"Failed to log download model completion on the server. "
                          f"Status code: {response.status}, content: {await notify_response.text()}")

            file_buffer.seek(0)
            with gzip.open(file_buffer, mode='rb') as f:
                global_state_dict = f.read()
        else:
            print(f"Failed to download the global model. Status code: {response.status}, content: {await response.text()}")

    return global_state_dict, configs

async def upload_local_model(session, local_state_dict, client_id):
    buffer = BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode='wb') as f:
        f.write(local_state_dict)
    buffer.seek(0)

    form_data = aiohttp.FormData()
    form_data.add_field('file', buffer, filename=f'{client_id}.pth.gz', content_type='application/gzip')

    async with session.post(UPLOAD_URL, data=form_data) as response:
        if response.status == 200:
            print("Model uploaded successfully.")
        else:
            print(f"Failed to upload the model. Status code: {response.status}, content: {await response.text()}")

async def handle_server(reader, writer, client_id, session):
    global dataset_cache
    local_model = get_model(config['dataset']['dataname'], config['dataset']['modelname'])
    original_shape_dict = {key: value.shape for key, value in local_model.state_dict().items()}

    while True:
        try:
            message = (await reader.read(1024)).decode('utf-8')
            if message == 'train':
                print(f"{client_id}: Received activating request from server")
                writer.write("train_ack".encode('utf-8'))
                await writer.drain()
                await asyncio.sleep(0.1)
                client_dataset = dataset_cache.get(client_id)
                if not client_dataset:
                    download_success = False
                    while not download_success:
                        try:
                            client_dataset = await download_dataset(session, client_id)
                            dataset_cache[client_id] = client_dataset
                            download_success = True
                        except Exception as e:
                            print(f"Error downloading client dataset: {e}")
                            await asyncio.sleep(1)
                else:
                    print(f"* {client_id} Local dataset is available")

                download_success = False
                local_configs = None
                while not download_success:
                    try:
                        global_state_dict, local_configs = await download_model_config(session, client_id)
                        global_state_dict = decompress_model(global_state_dict, original_shape_dict)
                        local_model.load_state_dict(global_state_dict)
                        download_success = True
                    except Exception as e:
                        print(f"Error downloading global model: {e}")
                        await asyncio.sleep(1)

                await train_model(client_id, local_model, client_dataset)
                local_state_dict = compress_model_untarget(local_model.state_dict(), local_configs['upload_ratio'])
                await upload_local_model(session, local_state_dict, client_id)
                writer.write("completed".encode('utf-8'))
                await writer.drain()
            elif message == 'waiting':
                print(f"{client_id}: now waiting")
            elif message == 'available':
                print(f"{client_id}: now available")

        except Exception as e:
            print(f"Error in handle_server: {e}")
            writer.close()
            await writer.wait_closed()
            break

async def train_model(client_id, local_model, client_dataset):
    data_train = client_dataset["dataloader_train"]
    # based on local dataset to perform local training

async def simulate_client():
    reader, writer = await asyncio.open_connection(SERVER_HOST, SERVER_COMM_PORT)
    client_id = (await reader.read(1024)).decode('utf-8')
    print(f"Client connected with server-assigned ID: {client_id}")
    writer.write("available".encode('utf-8'))
    await writer.drain()
    async with aiohttp.ClientSession() as session:
        await handle_server(reader, writer, client_id, session)

async def main(num_clients):
    tasks = [simulate_client() for _ in range(num_clients)]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    num_clients = config['dataset']['num_client']
    asyncio.run(main(num_clients))
