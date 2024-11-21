import torch
import os

# Replace with your path
current_path = "/home/tacucumides/storage/nbfnet-experiments/KnowledgeGraphCompletionV2/FB15k237-2/NBFNet/data"

def load_tensor(number, script_dir = current_path):
    # Get the directory of the current Python script

    file_path = os.path.join(script_dir, f"{number}.pt")
    if os.path.exists(file_path):
        return torch.load(file_path)
    else:
        print(f"Tensor file {number}.pt does not exist.")
        return None

def save_tensor(number, relation, path=current_path):
    pred_filename = os.path.join(path, f'{number}.pt')
    torch.save(relation, pred_filename)

def transform_tensor(n_rel, tensor, graph=):
    print("Transformando relacion", n_rel)
    links = get_links(n_rel)
    print("Reemplazando", len(links), " entradas")
    print("Links:", links)
    replace_in_matrix(links, tensor)

    return tensor

def get_links(relation, txt_file="train.txt"):
    """ returns list of (u,v) in train for a particular relation """
    # open txt file
    result = []
    file_path = os.path.join(current_path, txt_file)
    with open(file_path, 'r') as file:
        for line in file:
            values = line.strip().split()
            a, b, c = map(int, values)  
            if b == relation:
                result.append((a, c))
            elif b%237 == relation:
                result.append((c, a))
    return result

def replace_in_matrix(link_list, tensor):
    """ receive list of tuples [(u,v)] and replaces it in matrix"""
    for u,v in link_list:
        tensor[u][v] = 1


if __name__ == "__main__":
    for fil in range(237):
        print("Relation", fil)
        relation = load_tensor(fil)
        #relation = relation.float()
        relation = transform_tensor(fil, relation)
        save_tensor(fil, relation)

        if False:
            # Sample a subset of the data for quartile estimation
            sample_size = 1000  # You can adjust the sample size based on your memory constraints
            sample_indices = torch.randint(0, relation.size(0), (sample_size,))
            sampled_data = relation[sample_indices]

            # Calculate min, max, and quartiles on the sampled data
            min_value = torch.min(sampled_data)
            max_value = torch.max(sampled_data)
            quartiles = torch.quantile(sampled_data, torch.tensor([0.25, 0.5, 0.75]))

            print("Minimum value:", min_value.item())
            print("Maximum value:", max_value.item())
            print("25th percentile:", quartiles[0].item())
            print("50th percentile (median):", quartiles[1].item())
            print("75th percentile:", quartiles[2].item())
            print("########################################################")