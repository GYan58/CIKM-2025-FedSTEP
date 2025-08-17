from Utils import *

class TraditionalDataset:
    def __init__(self, dataset_name, root='./data'):
        self.dataset_name = dataset_name.lower()
        self.root = root
        self.transform = self.get_transform()
        self.train_data, self.test_data = self.load_dataset()

    def get_transform(self):
        if self.dataset_name in ['cifar10']:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
            ])
        elif self.dataset_name in ['cifar100']:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        elif self.dataset_name in ['fashionmnist']:
            return transforms.Compose([
                transforms.Resize(32),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.1307], std=[0.3081])
            ])
        elif self.dataset_name in ['mnist']:
            return transforms.Compose([
                transforms.Resize(32),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.06078], std=[0.1957])
            ])
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def load_dataset(self):
        if self.dataset_name == 'cifar10':
            train_dataset = CIFAR10(root=self.root, train=True, download=True, transform=self.transform)
            test_dataset = CIFAR10(root=self.root, train=False, download=True, transform=self.transform)
        elif self.dataset_name == 'cifar100':
            train_dataset = CIFAR100(root=self.root, train=True, download=True, transform=self.transform)
            test_dataset = CIFAR100(root=self.root, train=False, download=True, transform=self.transform)
        elif self.dataset_name == 'fashionmnist':
            train_dataset = FashionMNIST(root=self.root, train=True, download=True, transform=self.transform)
            test_dataset = FashionMNIST(root=self.root, train=False, download=True, transform=self.transform)
        elif self.dataset_name == 'mnist':
            train_dataset = MNIST(root=self.root, train=True, download=True, transform=self.transform)
            test_dataset = MNIST(root=self.root, train=False, download=True, transform=self.transform)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        return train_dataset, test_dataset

    def get_data(self):
        trainX = [data[0] for data in self.train_data]
        trainY = [data[1] for data in self.train_data]
        testX = [data[0] for data in self.test_data]
        testY = [data[1] for data in self.test_data]

        trainX = torch.stack(trainX)
        trainY = torch.tensor(trainY)
        testX = torch.stack(testX)
        testY = torch.tensor(testY)

        return trainX, trainY, testX, testY

ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
NUM_LETTERS = len(ALL_LETTERS)

class ShakespeareDataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.train_data_dir = os.path.join(root_dir, 'data', 'train')
        self.test_data_dir = os.path.join(root_dir, 'data', 'test')
        self.clients = []
        self.train_data = defaultdict(lambda: None)
        self.test_data = defaultdict(lambda: None)
        self.trainX, self.trainY, self.testX, self.testY = [], [], [], []
        self._load_data()

    def _word_to_indices(self, word):
        return torch.LongTensor(np.array([ALL_LETTERS.find(c) for c in word]))

    def _letter_to_vec(self, letter):
        return ALL_LETTERS.find(letter)

    def _read_dir(self, data_dir):
        clients = []
        data = defaultdict(lambda: {'x': [], 'y': []})

        files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
        for file in files:
            file_path = os.path.join(data_dir, file)
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
            clients.extend(cdata['users'])
            for user in cdata['user_data']:
                data[user]['x'].extend(cdata['user_data'][user]['x'])
                data[user]['y'].extend(cdata['user_data'][user]['y'])

        return list(sorted(data.keys())), data

    def _read_data(self):
        train_clients, train_data = self._read_dir(self.train_data_dir)
        test_clients, test_data = self._read_dir(self.test_data_dir)

        assert train_clients == test_clients

        self.clients = train_clients
        self.train_data = train_data
        self.test_data = test_data

    def _process_data(self):
        for key in self.clients:
            self.trainX += self.train_data[key]["x"]
            self.trainY += self.train_data[key]["y"]
            self.testX += self.test_data[key]["x"]
            self.testY += self.test_data[key]["y"]

        self.trainX = [self._word_to_indices(seq) for seq in self.trainX]
        self.trainY = torch.tensor([self._letter_to_vec(label) for label in self.trainY], dtype=torch.long)
        self.testX = [self._word_to_indices(seq) for seq in self.testX]
        self.testY = torch.tensor([self._letter_to_vec(label) for label in self.testY], dtype=torch.long)

    def _load_data(self):
        self._read_data()
        self._process_data()

    def get_data(self):
        trainX = torch.stack(self.trainX)
        trainY = torch.tensor(self.trainY)
        testX = torch.stack(self.testX)
        testY = torch.tensor(self.testY)
        return trainX, trainY, testX, testY


class HARBoxDataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.clients = []
        self.train_data = defaultdict(lambda: {'x': [], 'y': []})
        self.test_data = defaultdict(lambda: {'x': [], 'y': []})
        self.trainX, self.trainY, self.testX, self.testY = [], [], [], []
        self._read_data()
        self._process_data()

    def _load_data(self, user_id):
        coll_class = []
        coll_label = []
        total_class = 0
        class_set = ['Call', 'Hop', 'typing', 'Walk', 'Wave']
        dimension_of_feature = 900

        for class_id in range(len(class_set)):
            read_path = os.path.join(self.root_dir, str(user_id), f"{class_set[class_id]}_train.txt")
            if os.path.exists(read_path):
                temp_original_data = np.loadtxt(read_path)
                temp_reshape = temp_original_data.reshape(-1, 100, 10)
                temp_coll = temp_reshape[:, :, 1:10].reshape(-1, dimension_of_feature)
                count_img = temp_coll.shape[0]
                temp_label = class_id * np.ones(count_img)
                coll_class.extend(temp_coll)
                coll_label.extend(temp_label)
                total_class += 1

        coll_class = np.array(coll_class)
        coll_label = np.array(coll_label)
        return coll_class, coll_label

    def _read_data(self):
        num_of_total_users = 120

        for current_user_id in range(1, num_of_total_users + 1):
            x_coll, y_coll = self._load_data(current_user_id)
            Tsize = int(len(x_coll) * 0.2) + 1
            x_train, x_test, y_train, y_test = train_test_split(x_coll, y_coll, test_size=Tsize, random_state=0)
            self.train_data[current_user_id]['x'].extend(x_train)
            self.train_data[current_user_id]['y'].extend(y_train)
            self.test_data[current_user_id]['x'].extend(x_test)
            self.test_data[current_user_id]['y'].extend(y_test)

        self.clients = list(range(1, num_of_total_users + 1))

    def _process_data(self):
        for key in self.clients:
            self.trainX += self.train_data[key]['x']
            self.trainY += self.train_data[key]['y']
            self.testX += self.test_data[key]['x']
            self.testY += self.test_data[key]['y']

        self.trainX = np.array(self.trainX, dtype=float)
        self.trainY = np.array(self.trainY, dtype=int)
        self.testX = np.array(self.testX, dtype=float)
        self.testY = np.array(self.testY, dtype=int)

    def get_data(self):
        trainX = torch.tensor(self.trainX, dtype=torch.float32)
        trainY = torch.tensor(self.trainY, dtype=torch.long)
        testX = torch.tensor(self.testX, dtype=torch.float32)
        testY = torch.tensor(self.testY, dtype=torch.long)
        return trainX, trainY, testX, testY


class DirichletDataSplitter:
    def __init__(self, dataset, alpha=0.5, batch_size=32, num_workers=1):
        self.dataset = dataset
        self.alpha = alpha
        self.batch_size = batch_size
        self.num_workers = num_workers

    def split_data_dirichlet(self, label_distribution, Y, N):
        num_classes = len(np.unique(Y))
        class_indices = [np.where(Y == y)[0] for y in range(num_classes)]
        client_indices = [[] for _ in range(N)]
        for c, fracs in zip(class_indices, label_distribution):
            for i, idx in enumerate(np.split(c, (np.cumsum(fracs[:-1]) * len(c)).astype(int))):
                client_indices[i] += [idx]
        return [np.concatenate(idx) for idx in client_indices]

    def allocate_random_samples(self, train_indices, test_indices, num_samples=20):
        non_empty_clients = [i for i in range(len(train_indices)) if
                             len(train_indices[i]) > num_samples and len(test_indices[i]) > num_samples]
        random_client = np.random.choice(non_empty_clients)
        train_select_length = int(np.ceil(len(train_indices[random_client]) / 2))
        test_select_length = int(np.ceil(len(test_indices[random_client]) / 2))
        random_train_indices = np.random.choice(train_indices[random_client], train_select_length, replace=False)
        random_test_indices = np.random.choice(test_indices[random_client], test_select_length, replace=False)
        return random_train_indices, random_test_indices

    def get_dataloaders(self, N):
        trainX, trainY, testX, testY = self.dataset.get_data()

        print(f'trainX shape: {trainX.shape}, trainY shape: {trainY.shape}')
        print(f'testX shape: {testX.shape}, testY shape: {testY.shape}')

        label_distribution = np.random.dirichlet([self.alpha] * N, len(np.unique(trainY.numpy())))
        train_indices = self.split_data_dirichlet(label_distribution, trainY.numpy(), N)
        test_indices = self.split_data_dirichlet(label_distribution, testY.numpy(), N)

        for i in range(N):
            if len(train_indices[i]) < 10 or len(test_indices[i]) < 10:
                train_indices[i], test_indices[i] = self.allocate_random_samples(train_indices, test_indices)

        client_train_loaders = []
        client_test_loaders = []
        for i in range(N):
            client_train_X = trainX[train_indices[i]]
            client_train_Y = trainY[train_indices[i]]
            client_test_X = testX[test_indices[i]]
            client_test_Y = testY[test_indices[i]]

            client_train_dataset = TensorDataset(client_train_X, client_train_Y)
            client_test_dataset = TensorDataset(client_test_X, client_test_Y)

            client_train_loader = DataLoader(client_train_dataset, batch_size=self.batch_size, shuffle=True,
                                             num_workers=self.num_workers)
            client_test_loader = DataLoader(client_test_dataset, batch_size=self.batch_size, shuffle=False,
                                            num_workers=self.num_workers)

            client_train_loaders.append(client_train_loader)
            client_test_loaders.append(client_test_loader)

        overall_train_dataset = TensorDataset(trainX, trainY)
        overall_test_dataset = TensorDataset(testX, testY)
        overall_train_loader = DataLoader(overall_train_dataset, batch_size=self.batch_size, shuffle=True,
                                          num_workers=self.num_workers)
        overall_test_loader = DataLoader(overall_test_dataset, batch_size=self.batch_size, shuffle=False,
                                         num_workers=self.num_workers)

        return overall_train_loader, overall_test_loader, client_train_loaders, client_test_loaders

def get_dataloaders(dataset_name, num_clients, alpha, batch_size):
    dataset_name = dataset_name.lower()
    if dataset_name not in ["harbox", "shakespeare"]:
        dataset = TraditionalDataset(dataset_name)
    elif dataset_name == "shakespeare":
        dataset = ShakespeareDataset("Shakespeare_root_dir")
    elif dataset_name == "harbox":
        dataset = HARBoxDataset("HARBox_root_dir")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    splitter = DirichletDataSplitter(dataset, alpha=alpha, batch_size=batch_size)
    return splitter.get_dataloaders(num_clients)
