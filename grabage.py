import torch
import random
import json


def create_shuffled_tensor_and_save(file_path, N=1000):
    # Generate a tensor with torch.arange(N)
    tensor = torch.arange(N)

    # Convert tensor to a list for shuffling
    tensor_list = tensor.tolist()

    # Shuffle the list
    random.shuffle(tensor_list)

    # Save the shuffled list as a JSON file
    with open(file_path, 'w') as json_file:
        json.dump(tensor_list, json_file)
    return tensor_list
file_path = './source_dataset_training/zinc12k_indices.json'
z = create_shuffled_tensor_and_save(N=1000, file_path=file_path)

file_path = './source_dataset_training/cifar10_indices.json'
z = create_shuffled_tensor_and_save(N=10000, file_path=file_path)
print("Asf")
