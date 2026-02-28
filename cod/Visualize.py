import cv2 as cv
import os
import numpy as np
import pdb
import ntpath
import glob
from Parameters import *
from collections import defaultdict


def show_detections_without_ground_truth(detections, scores, file_names, params: Parameters):
    """
    Afiseaza si salveaza imaginile adnotate.
    detections: numpy array de dimensiune NX4, unde N este numarul de detectii pentru toate imaginile.
    detections[i, :] = [x_min, y_min, x_max, y_max]
    scores: numpy array de dimensiune N, scorurile pentru toate detectiile pentru toate imaginile.
    file_names: numpy array de dimensiune N, pentru fiecare detectie trebuie sa salvam numele imaginii.
    (doar numele, nu toata calea).
    """
    test_images_path = os.path.join(params.dir_test_examples, '*.jpg')
    test_files = glob.glob(test_images_path)

    for test_file in test_files:
        image = cv.imread(test_file)
        short_file_name = ntpath.basename(test_file)
        indices_detections_current_image = np.where(file_names == short_file_name)
        current_detections = detections[indices_detections_current_image]
        current_scores = scores[indices_detections_current_image]

        for idx, detection in enumerate(current_detections):
            cv.rectangle(image, (detection[0], detection[1]), (detection[2], detection[3]), (0, 0, 255), thickness=1)
            cv.putText(image, 'score:' + str(current_scores[idx])[:4], (detection[0], detection[1]),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv.imwrite(os.path.join(params.dir_save_files_task1, "detections_" + short_file_name), image)
        print('Apasa orice tasta pentru a continua...')
        cv.imshow('image', np.uint8(image))
        cv.waitKey(0)


def show_detections_with_ground_truth(detections, scores, file_names, params: Parameters):
    """
    Afiseaza si salveaza imaginile adnotate. Deseneaza bounding box-urile prezice si cele corecte.
    detections: numpy array de dimensiune NX4, unde N este numarul de detectii pentru toate imaginile.
    detections[i, :] = [x_min, y_min, x_max, y_max]
    scores: numpy array de dimensiune N, scorurile pentru toate detectiile pentru toate imaginile.
    file_names: numpy array de dimensiune N, pentru fiecare detectie trebuie sa salvam numele imaginii.
    (doar numele, nu toata calea).
    """

    ground_truth_bboxes = np.loadtxt(params.path_annotations, dtype='str')
    test_images_path = os.path.join(params.dir_test_examples, '*.jpg')
    test_files = glob.glob(test_images_path)

    out_dir = '../save/task1/detections'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for test_file in test_files:
        image = cv.imread(test_file)
        short_file_name = ntpath.basename(test_file)
        indices_detections_current_image = np.where(file_names == short_file_name)
        current_detections = detections[indices_detections_current_image]
        current_scores = scores[indices_detections_current_image]

        for idx, detection in enumerate(current_detections):
            cv.rectangle(image, (detection[0], detection[1]), (detection[2], detection[3]), (0, 0, 255), thickness=1)
            cv.putText(image, 'score:' + str(current_scores[idx])[:4], (detection[0], detection[1]),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        annotations = ground_truth_bboxes[ground_truth_bboxes[:, 0] == short_file_name]

        # show ground truth bboxes
        for detection in annotations:
            cv.rectangle(image, (int(detection[1]), int(detection[2])), (int(detection[3]), int(detection[4])), (0, 255, 0), thickness=1)

        cv.imwrite(os.path.join(out_dir, "detections_" + short_file_name), image)
        print('Apasa orice tasta pentru a continua...')
        cv.imshow('image', np.uint8(image))
        cv.waitKey(0)

def build_detection_dict(detections, filenames, scores):
    """
    Returnează dict: filename -> [(bbox, score), ...]
    bbox este o listă [x1,y1,x2,y2] (int).
    """
    d = defaultdict(list)

    filenames = filenames.astype(str)
    for det, sc, fname in zip(detections, scores, filenames):
        bbox = [int(det[0]), int(det[1]), int(det[2]), int(det[3])]
        d[fname].append((bbox, sc))

    return dict(d)

def show_detections_with_ground_truth_task2(params):
    daphne_detections = np.load('../save/task2/detections_daphne.npy')
    fred_detections   = np.load('../save/task2/detections_fred.npy')
    velma_detections  = np.load('../save/task2/detections_velma.npy')
    shaggy_detections = np.load('../save/task2/detections_shaggy.npy')
    daphne_detections = daphne_detections.astype(np.int32)
    fred_detections = fred_detections.astype(np.int32)
    velma_detections = velma_detections.astype(np.int32)
    shaggy_detections = shaggy_detections.astype(np.int32)

    filenames_daphne = np.load('../save/task2/filenames_daphne.npy').astype(str)
    filenames_fred   = np.load('../save/task2/filenames_fred.npy').astype(str)
    filenames_velma  = np.load('../save/task2/filenames_velma.npy').astype(str)
    filenames_shaggy = np.load('../save/task2/filenames_shaggy.npy').astype(str)

    scores_daphne = np.load('../save/task2/scores_daphne.npy').astype(np.float32)
    scores_fred   = np.load('../save/task2/scores_fred.npy').astype(np.float32)
    scores_velma  = np.load('../save/task2/scores_velma.npy').astype(np.float32)
    scores_shaggy = np.load('../save/task2/scores_shaggy.npy').astype(np.float32)

    # dict: filename -> [(bbox, score), ...]
    daphne_dict = build_detection_dict(daphne_detections, filenames_daphne, scores_daphne)
    fred_dict   = build_detection_dict(fred_detections,   filenames_fred,   scores_fred)
    velma_dict  = build_detection_dict(velma_detections,  filenames_velma,  scores_velma)
    shaggy_dict = build_detection_dict(shaggy_detections, filenames_shaggy, scores_shaggy)

    out_dir = '../save/task2/detections'
    os.makedirs(out_dir, exist_ok=True)

    # culori BGR (OpenCV)
    COLORS = {
        "fred":   (63, 154, 174),     # albastru
        "daphne": (229, 186, 65),   # galben
        "shaggy": (92, 111, 43),     # verde
        "velma":  (110, 2, 111)    # violet
    }

    def draw_from_dict(img, img_name, det_dict, color, label=None):
        """Desenează toate detecțiile din det_dict[img_name] dacă există."""
        if img_name not in det_dict:
            return

        for bbox, score in det_dict[img_name]:
            x1, y1, x2, y2 = map(int, bbox)
            cv.rectangle(img, (x1, y1), (x2, y2), color, thickness=4)

            txt = f"{score:.2f}" if label is None else f"{label} {score:.2f}"
            # text deasupra box-ului (dacă nu încape, îl pune sub)
            tx = x1
            ty = y1 - 5
            if ty < 12:
                ty = y1 + 15

            cv.putText(img, txt, (tx, ty), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv.LINE_AA)

    test_images_path = os.path.join(params.dir_test_examples, '*.jpg')
    test_files = glob.glob(test_images_path)

    for test_file in test_files:
        img_name = os.path.basename(test_file)
        img = cv.imread(test_file)
        if img is None:
            continue

        # desenează pe rând pentru fiecare personaj
        draw_from_dict(img, img_name, fred_dict,   COLORS["fred"], label='fred')
        draw_from_dict(img, img_name, daphne_dict, COLORS["daphne"], label = 'daphne')
        draw_from_dict(img, img_name, shaggy_dict, COLORS["shaggy"], label = 'shaggy')
        draw_from_dict(img, img_name, velma_dict,  COLORS["velma"], label = 'velma')

        # salvează + afișează
        out_path = os.path.join(out_dir, img_name)
        cv.imwrite(out_path, img)

        cv.imshow(f'detections_task2_{img_name}', img)
        cv.waitKey(0)

    cv.destroyAllWindows()