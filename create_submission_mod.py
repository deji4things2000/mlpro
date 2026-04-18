from torch import load, save
from torch.nn import Softmax
from os.path import exists
from zipfile import ZipFile
from sys import argv
from cnn_categorization_base import cnn_categorization_base
from cnn_categorization_improved import cnn_categorization_improved
import torch


def create_submission(model_type):
    """
    Evaluates a trained model on the test and validation sets and creates an archive containing the evaluation results
    and the model.

    Arguments
    ---------
    model_type: (string) specifies the model {base, improved}

    """

    data_path = "{}_image_categorization_dataset.pt".format(model_type)
    model_path = "{}-model.pt".format(model_type)

    assert exists(data_path), f"The data file {data_path} does not exist"
    assert exists(model_path), f"The trained model {model_path} does not exist"

    print(f"Loading dataset from {data_path}...")
    dataset = load(data_path)
    # the relevant data sets
    data_te = dataset["data_te"]
    sets_tr = dataset["sets_tr"]
    data_val = dataset["data_tr"]
    data_val = data_val[sets_tr == 2]
    print(f"Test set size: {data_te.shape}, Validation set size: {data_val.shape}")

    print(f"Loading model from {model_path}...")
    model_state = load(model_path)
    if model_type == 'base':
        model = cnn_categorization_base(model_state['specs'])
    else:
        model = cnn_categorization_improved(model_state['specs'])

    print(f"Loading model weights from {model_state['state']}...")
    model.load_state_dict(load(model_state['state']))
    model.eval()
    soft_max = Softmax(dim=1)

    # Process test set in batches
    batch_size = 128
    print(f"Processing test set in batches of {batch_size}...")
    prob_test_list = []
    with torch.no_grad():
        for i in range(0, len(data_te), batch_size):
            batch = data_te[i:i+batch_size]
            batch_prob = soft_max(model(batch).squeeze())
            prob_test_list.append(batch_prob)
            if (i // batch_size) % 10 == 0:
                print(f"  Processed {min(i+batch_size, len(data_te))}/{len(data_te)} test examples")
    
    prob_test = torch.cat(prob_test_list, dim=0)
    print(f"Test probabilities shape: {prob_test.size()}")
    assert prob_test.size() == (9600, 16), f"Expected the output of the test set to be of size (9600, 16) " \
                                           f"but was {prob_test.size()} instead"
    
    # Process validation set in batches
    print(f"Processing validation set in batches of {batch_size}...")
    prob_val_list = []
    with torch.no_grad():
        for i in range(0, len(data_val), batch_size):
            batch = data_val[i:i+batch_size]
            batch_prob = soft_max(model(batch).squeeze())
            prob_val_list.append(batch_prob)
            if (i // batch_size) % 10 == 0:
                print(f"  Processed {min(i+batch_size, len(data_val))}/{len(data_val)} validation examples")
    
    prob_val = torch.cat(prob_val_list, dim=0)
    print(f"Validation probabilities shape: {prob_val.size()}")
    assert prob_val.size() == (6400, 16), f"Expected the output of the validation set to be of size (6400, 16) " \
                                          f"but was {prob_val.size()} instead"

    output_name_zip = "./{}_categorization.zip".format(model_type)
    output_name_test = "./{}_testing.pt".format(model_type)
    output_name_val = "./{}_validation.pt".format(model_type)
    
    print(f"Saving {output_name_test}...")
    save(prob_test, output_name_test)
    print(f"Saving {output_name_val}...")
    save(prob_val, output_name_val)
    
    print(f"Creating {output_name_zip}...")
    with ZipFile(output_name_zip, 'w') as zipf:
        zipf.write(model_path)
        zipf.write(output_name_test)
        zipf.write(output_name_val)

        if model_type == "improved" and not exists("submission_details.txt"):
            raise FileNotFoundError("Please create a file submission_details.txt describing your improvements")
        if exists("submission_details.txt"):
            zipf.write("submission_details.txt")
            print("Added submission_details.txt")
    
    print(f"Successfully created {output_name_zip}")


if __name__ == '__main__':

    # change m_type to 'improved' for the improved task
    # You can also pass in an argument from the command line

    m_type = "base"
    try:
        m_type = argv[1]
    except IndexError:
        print("Setting model type to <base> since no type is given.")
        print("To create an archive for the improved model,"
              " execute <create_submission.py improved> at the command line")
    assert m_type in ("base", "improved"), f"Model type must be either base or improved. Found {m_type} instead"

    create_submission(m_type)