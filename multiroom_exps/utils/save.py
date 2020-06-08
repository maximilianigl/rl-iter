import csv
import json
import logging
import os
import sys

import torch
import utils
import pymongo
import gridfs


def get_model_path(model_dir, suffix=None):
    model_name = "model"+str(suffix)+".pt"
    return os.path.join(model_dir, model_name)

def load_model(model_dir):
    try:
        path1 = get_model_path(model_dir, 1)
        model1 = torch.load(path1)
        model1.eval()
    except:
        model1 = None

    # This should always exist as it's the training model!
    path2 = get_model_path(model_dir, 2)
    model2 = torch.load(path2)
    model2.eval()
    return model1, model2

def save_model(model1, model2, model_dir):
    if torch.cuda.is_available():
        model2.cpu()
        if model1 is not None:
            model1.cpu()

    if model1 is not None:
        path1 = get_model_path(model_dir, 1)
        utils.create_folders_if_necessary(path1)
        torch.save(model1, path1)
    if model2 is not None:
        path2 = get_model_path(model_dir, 2)
        utils.create_folders_if_necessary(path2)
        torch.save(model2, path2)

    if torch.cuda.is_available():
        model2.cuda()
        if model1 is not None:
            model1.cuda()

def save_model_to_db(model1, model2, model_dir, num_frames, _run):
    save_model(model1, model2, model_dir)
    if model1 is not None:
        path1 = get_model_path(model_dir, 1)
        _run.add_artifact(path1, name="pi_old", metadata={'num_frames': num_frames})
    if model2 is not None:
        path2 = get_model_path(model_dir, 2)
        _run.add_artifact(path2, name="pi_train", metadata={'num_frames': num_frames})

def save_status_to_db(status, model_dir, num_frames, _run):
    save_status(status, model_dir)
    path = get_status_path(model_dir)
    _run.add_artifact(path, name="status.json", metadata={'num_frames': num_frames})

def get_docs(url, db, col):
    client = pymongo.MongoClient(url, ssl=True)
    return client[db][col]

def get_file_id(doc, file_name):
    """
    Helper function to access data when MongoObserver is used.
    Go through all files in doc and return the id of the file with file_name.
    """
    r = list(filter(lambda dic: dic['name'] == file_name, doc['artifacts']))
    if len(r) == 0:
        raise KeyError("Artifact not found")
    assert len(r) == 1
    return r[0]['file_id']


def save_file_from_db(file_id, destination, db_uri, db_name):
    """
    Given a file_id (e.g. through get_file_id()) and a db_uri (a db connection string),
    save the corresponding file to `destination` (filename as string).
    """
    client = pymongo.MongoClient(db_uri, ssl=True)
    fs = gridfs.GridFSBucket(client[db_name])
    open_file = open(destination, 'wb+')
    fs.download_to_stream(file_id, open_file)

def load_status_and_model_from_db(url, db, model_dir, load_id):
    docs = get_docs(url, db, 'runs')
    projection = {"config": True, "_id": True, "artifacts": True}
    doc = docs.find({"_id": load_id}, projection)
    assert doc.count() == 1
    doc = doc.next()

    try:
        model1_id = get_file_id(doc, "pi_old")
        model1_path = get_model_path(model_dir, 1)
        save_file_from_db(model1_id, model1_path, url, db)
    except KeyError as e:
        # No pi_old has been saved to database
        # No need to to anything because load_model deals with it and returns None for model1
        pass

    model2_id = get_file_id(doc, "pi_train")
    model2_path = get_model_path(model_dir, 2)
    save_file_from_db(model2_id, model2_path, url, db)

    status_id = get_file_id(doc, "status.json")
    status_path = get_status_path(model_dir)
    save_file_from_db(status_id, status_path, url, db)

    model1, model2 = load_model(model_dir)
    status = load_status(model_dir)
    return model1, model2, status


def get_status_path(model_dir):
    return os.path.join(model_dir, "status.json")

def load_status(model_dir):
    path = get_status_path(model_dir)
    with open(path) as file:
        return json.load(file)

def save_status(status, model_dir):
    path = get_status_path(model_dir)
    utils.create_folders_if_necessary(path)
    with open(path, "w") as file:
        json.dump(status, file)

def get_log_path(model_dir):
    return os.path.join(model_dir, "log.txt")

def get_logger(model_dir):
    path = get_log_path(model_dir)
    utils.create_folders_if_necessary(path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(filename=path),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger()

def get_vocab_path(model_dir):
    return os.path.join(model_dir, "vocab.json")

def get_csv_path(model_dir):
    return os.path.join(model_dir, "log.csv")

def get_csv_writer(model_dir):
    csv_path = get_csv_path(model_dir)
    utils.create_folders_if_necessary(csv_path)
    csv_file = open(csv_path, "a")
    return csv_file, csv.writer(csv_file)