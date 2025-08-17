from Dataset import get_dataloaders, save_dataloader_info
from Utils import *

# Initialize Flask app
app = Flask(__name__)
UPLOAD_MODEL = config['paths']['upload_model']
UPLOAD_CONFIG = config['paths']['upload_config']
UPLOAD_DATASET = config['paths']['upload_dataset'] + '/' + config['dataset']['dataname'] + '-' + str(config['dataset']['alpha']) + "A"
os.makedirs(UPLOAD_MODEL, exist_ok=True)
os.makedirs(UPLOAD_CONFIG, exist_ok=True)
os.makedirs(UPLOAD_DATASET, exist_ok=True)

logging.basicConfig(
    filename=config['paths']['log_path_server_fedbridge'],
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

class FLEfficiencyMetric:
    def __init__(self, window_size=10):
        self.upload_data_window = deque(maxlen=window_size)
        self.upload_time_window = deque(maxlen=window_size)
        self.download_data_window = deque(maxlen=window_size)
        self.download_time_window = deque(maxlen=window_size)
        self.lock = threading.Lock()

    def log_upload(self, data_size, communication_time):
        with self.lock:
            self.upload_data_window.append(data_size)
            self.upload_time_window.append(communication_time)

    def log_download(self, data_size, communication_time):
        with self.lock:
            self.download_data_window.append(data_size)
            self.download_time_window.append(communication_time)

    def get_metrics(self):
        with self.lock:
            total_upload_data = sum(self.upload_data_window)
            total_upload_time = sum(self.upload_time_window)
            total_download_data = sum(self.download_data_window)
            total_download_time = sum(self.download_time_window)

            average_upload_time = total_upload_time / len(self.upload_time_window) if self.upload_time_window else 0
            upload_efficiency = total_upload_data / total_upload_time if total_upload_time else 0
            average_download_time = total_download_time / len(self.download_time_window) if self.download_time_window else 0
            download_efficiency = total_download_data / total_download_time if total_download_time else 0

            return {
                "total_upload_data_mb": total_upload_data,
                "total_upload_time_s": total_upload_time,
                "average_upload_time_per_round_s": average_upload_time,
                "upload_efficiency_mb_per_s": upload_efficiency,
                "total_download_data_mb": total_download_data,
                "total_download_time_s": total_download_time,
                "average_download_time_per_round_s": average_download_time,
                "download_efficiency_mb_per_s": download_efficiency
            }

FLMetric = FLEfficiencyMetric()

def log_metrics_periodically(interval=10):
    while True:
        time.sleep(interval)
        metrics_data = FLMetric.get_metrics()
        logging.info(f'Metrics: {metrics_data}')

threading.Thread(target=log_metrics_periodically, daemon=True).start()

if config['dataset']['split']:
    overall_train_loader, overall_test_loader, client_train_loaders, client_test_loaders = get_dataloaders(
        config['dataset']['dataname'],
        config['dataset']['num_client'],
        config['dataset']['alpha'],
        config['dataset']['batch_size']
    )

    for i in range(config['dataset']['num_client']):
        WPath = os.path.join(UPLOAD_DATASET, f'client_{i}.pkl')
        data_info = {
            'data_name': config['dataset']['dataname'],
            'client_id': f'client_{i}'
        }
        data_loader_train = client_train_loaders[i]
        data_loader_test = client_test_loaders[i]
        save_dataloader_info(data_info, data_loader_train, data_loader_test, WPath)

def is_valid_filename(filename):
    file_split = filename.split(".")
    if len(file_split) > 1 and file_split[-1] in ["gz", "pth", "json", "pkl"]:
        return True
    return False

@app.route('/upload', methods=['POST'])
def upload_file():
    start_time = time.time()

    if 'file' not in request.files:
        logging.warning('No file part in the request')
        return jsonify({'message': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        logging.warning('No selected file')
        return jsonify({'message': 'No selected file'}), 400
    if not is_valid_filename(file.filename):
        logging.warning('Invalid file name')
        return jsonify({'message': 'Invalid file name'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_MODEL, filename)
    file.save(file_path)

    upload_size = os.path.getsize(file_path) / (1024 * 1024)

    end_time = time.time()
    communication_time = end_time - start_time
    FLMetric.log_upload(upload_size, communication_time)

    logging.info(f'Uploaded file: {file.filename}, Size: {upload_size:.2f} MB, Time: {communication_time:.2f} s')
    return jsonify({'message': 'File successfully uploaded'}), 200

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    if not is_valid_filename(filename):
        logging.warning('Invalid file name for download')
        return jsonify({'message': f'Invalid file name:{filename}'}), 400

    file_path = None
    if ".pth" in filename:
        file_path = os.path.join(UPLOAD_MODEL, filename)
    elif ".pkl" in filename:
        file_path = os.path.join(UPLOAD_DATASET, filename)
    elif ".json" in filename:
        file_path = os.path.join(UPLOAD_CONFIG, filename)

    if not os.path.exists(file_path):
        logging.warning(f'File not found: {filename}')
        abort(404, description=f"File not found: {filename}")

    if ".pth" in filename:
        return send_from_directory(UPLOAD_MODEL, filename, as_attachment=True)
    elif ".pkl" in filename:
        return send_from_directory(UPLOAD_DATASET, filename, as_attachment=True)
    elif ".json" in filename:
        with open(file_path, 'r') as f:
            config_data = json.load(f)
        return jsonify(config_data)

@app.route('/download_complete', methods=['POST'])
def download_complete():
    data = request.get_json()
    if not data or 'filename' not in data or 'start_time' not in data:
        logging.warning('Invalid download completion request')
        return jsonify({'message': 'Invalid request'}), 400

    filename = data['filename']
    start_time = data['start_time']
    end_time = time.time()
    communication_time = end_time - start_time

    file_path = None
    if ".pth" in filename:
        file_path = os.path.join(UPLOAD_MODEL, filename)
    elif ".pkl" in filename:
        file_path = os.path.join(UPLOAD_DATASET, filename)
    if not os.path.exists(file_path):
        logging.warning(f'File not found for download completion: {filename}')
        return jsonify({'message': 'File not found'}), 404

    file_size = os.path.getsize(file_path) / (1024 * 1024)
    FLMetric.log_download(file_size, communication_time)

    logging.info(f'Completed download: {filename}, Size: {file_size:.2f} MB, Time: {communication_time:.2f} s')
    return jsonify({'message': 'Download completion logged'}), 200

@app.route('/metrics', methods=['GET'])
def metrics():
    metrics_data = FLMetric.get_metrics()
    logging.info('Retrieved metrics')
    return jsonify(metrics_data)

if __name__ == '__main__':
    app.run(host=config['server']['host'], port=config['server']['trans_port'])