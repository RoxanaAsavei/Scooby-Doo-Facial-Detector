from Parameters import *
from FacialDetector import *
import pdb
from Visualize import *
from FacialClassifier import *

params: Parameters = Parameters()
params.dim_window = 64  # exemplele pozitive (fete de oameni cropate) au 64x64 pixeli
params.overlap = 0.3
params.number_positive_examples = 6547  # numarul exemplelor pozitive
params.number_negative_examples = 80000 # numarul exemplelor negative

params.threshold = 0.8 # toate ferestrele cu scorul >= threshold si maxime locale devin detectii
params.has_annotations = False

params.use_hard_mining = False  # (optional)antrenare cu exemple puternic negative
params.use_flip_images = True  # adauga imaginile cu fete oglindite
params.augument_positives = True

if params.augument_positives: # pentru augumentari suplimentare
    params.number_positive_examples *= 4

facial_detector: FacialDetector = FacialDetector(params)

# Pasii 1+2+3. Incarcam exemplele pozitive (cropate) si exemple negative generate
# verificam daca sunt deja existente
"""
# ------------ pentru crearea, antrenarea modelelor ------------------
positive_features_path = '../save/pos_pixel_values.npy'
if os.path.exists(positive_features_path):
    positive_features = np.load(positive_features_path)
    print('Am incarcat exemplele pozitive')
else:
    print('Construim descriptorii pentru exemplele pozitive:')
    positive_features = facial_detector.get_positive_images()
    np.save(positive_features_path, positive_features)
    print('Am salvat exemplele pozitive in fisierul %s' % positive_features_path)

# # exemple negative
negative_features_path = '../save/neg_pixel_values.npy'
if os.path.exists(negative_features_path):
    negative_features = np.load(negative_features_path)
    print('Am incarcat exemplele negative')
else:
    print('Construim descriptorii pentru exemplele negative:')
    negative_features = facial_detector.get_negative_images()
    np.save(negative_features_path, negative_features)
    print('Am salvat exemplele negative in fisierul %s' % negative_features_path)

training_examples = np.concatenate((np.squeeze(positive_features), np.squeeze(negative_features)), axis=0)

train_labels = np.concatenate((np.ones(positive_features.shape[0]), np.zeros(negative_features.shape[0])))
facial_detector.train_classifier(training_examples, train_labels, 'model_first.h5', 'none')

negative_features_hm_path = '../save/neg_pixel_values_hm.npy'
if os.path.exists(negative_features_hm_path):
    negative_features_hm = np.load(negative_features_hm_path)
    print('Am incarcat exemplele hard negative')
else:
    print('Construim descriptorii pentru exemplele hard negative:')
    negative_features_hm = facial_detector.get_negative_images_hm('../examples/hardNegatives')
    np.save(negative_features_hm_path, negative_features_hm)
    print('Am salvat exemplele hard negative in fisierul %s' % negative_features_hm_path)


training_examples_hm = np.concatenate(
    (np.squeeze(positive_features), np.squeeze(negative_features), np.squeeze(negative_features_hm)),
    axis=0 
)

train_labels_hm = np.concatenate(
    (np.ones(positive_features.shape[0]), np.zeros(negative_features.shape[0] + negative_features_hm.shape[0]))
)

# antrenam modelul pe datsetul cu hard negatives
facial_detector.train_classifier(training_examples_hm, train_labels_hm, 'model_post_hm.h5', 'model_first.h5')
"""



detections, scores, file_names = facial_detector.run()
# salvam rezultatele
detections_all_faces_path = '../save/task1/detections_all_faces.npy'
scores_all_faces_path = '../save/task1/scores_all_faces.npy'
file_names_all_faces_path = '../save/task1/file_names_all_faces.npy'

np.save(detections_all_faces_path, detections)
np.save(scores_all_faces_path, scores)
np.save(file_names_all_faces_path, file_names)

if params.has_annotations:
    facial_detector.eval_detections('../validare/task1_gt_validare.txt', detections, scores, file_names)
    show_detections_with_ground_truth(detections, scores, file_names, params)
else:
    show_detections_without_ground_truth(detections, scores, file_names, params)

# clasficare
detections = detections.astype(np.int32)
facial_classifier: FacialClassifier = FacialClassifier(params)
facial_classifier.classify(detections, file_names)
# show_detections_with_ground_truth_task2(params) comentat