from Parameters import *
from Utils import *
import numpy as np
import keras
import keras.models
import matplotlib.pyplot as plt
import glob
import cv2 as cv
from keras.layers import (
    Conv2D, MaxPooling2D,
    Dense, Dropout, BatchNormalization,
    GlobalAveragePooling2D, ReLU
)
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from pathlib import Path
import timeit

class FacialClassifier:
    def __init__(self, params: Parameters):
        self.params = params
        self.model = None

    def extract_character_features(self, character):
        upload_filepath = f'../examples/positiveExamples/{character}'
        images_path = os.path.join(upload_filepath, '*.jpg')
        files = glob.glob(images_path)
        num_images = len(files)
        print(f'Extragem features pentru {num_images} imagini cu {character}')
        character_images = []
        for i in range(num_images):
            img = cv.imread(files[i]) # BGR image
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0 # normalizam imaginea originala
            character_images.append(img)
            # facem flip orizontal
            flipped_horiz = np.fliplr(img)
            character_images.append(flipped_horiz)
            if self.params.augument_positives:
                character_images.append(aug_blur(img))
                character_images.append(aug_rotate(img, 5))
                character_images.append(aug_brightness_contrast(img))
                character_images.append(aug_shift(img))

        character_images = np.array(character_images)
        return character_images
    
    def load_pixel_values(self, character):
        save_filepath = f'../save/task2/{character}_pixel_values.npy'
        if os.path.exists(save_filepath):
            character_features = np.load(save_filepath)
            print(f'Am incarcat exemplele pentru {character}')
        else: # nu au mai fost procesate img pentru acest personaj
            print(f'Incepem extragere features pentru {character}')
            character_features = self.extract_character_features(character)
            np.save(save_filepath, character_features)
            print(f'Au fost salvate features pentru {character}')
        return character_features
    


    def build_dataset(self, no_random_neg = 1000):
        daphne_features = self.load_pixel_values('daphne') # 0
        fred_features = self.load_pixel_values('fred') # 1
        velma_features = self.load_pixel_values('velma') # 2
        shaggy_features = self.load_pixel_values('shaggy') # 3 
        unknown_features = self.load_pixel_values('unknown') # 4
        negative_features = np.load('../save/neg_pixel_values.npy') # 4  
        negative_features_hm = np.load('../save/neg_pixel_values_hm.npy') # 4

        # luam random cam 1000 de negative_features (care au fost alese aleator)
        N = negative_features.shape[0]
        idx = np.random.choice(N, size=no_random_neg, replace=False)
        negative_features = negative_features[idx]

        training_data = np.concatenate(
            (np.squeeze(daphne_features), np.squeeze(fred_features), np.squeeze(velma_features),
             np.squeeze(shaggy_features), np.squeeze(unknown_features), np.squeeze(negative_features),
             np.squeeze(negative_features_hm)),
            axis = 0
        )
        training_labels = np.concatenate(
            (np.full(len(daphne_features), 0, dtype=np.int32),
             np.full(len(fred_features), 1, dtype=np.int32),
             np.full(len(velma_features), 2, dtype=np.int32),
             np.full(len(shaggy_features), 3, dtype=np.int32),
             np.full(len(unknown_features) + len(negative_features_hm) + len(negative_features), 4, dtype=np.int32)),
            axis = 0
        )

        return training_data, training_labels
    
    def build_model(self, num_classes = 5):
        model = Sequential()
        model.add(Conv2D(16, 3, padding="same", use_bias=False, input_shape=(64, 64, 3)))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(MaxPooling2D())
        model.add(Dropout(0.10))

        model.add(Conv2D(32, 3, padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(MaxPooling2D())
        model.add(Dropout(0.15))

        model.add(Conv2D(64, 3, padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(MaxPooling2D())
        model.add(Dropout(0.20))

        model.add(GlobalAveragePooling2D())
        model.add(Dense(64, use_bias=False))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Dropout(0.30))

        model.add(Dense(num_classes, activation="softmax"))

        return model
    
    def train_classifier(self, training_examples, train_labels, save_filename, prev_model=None):

        prev_model_path = f'../save/task2/{prev_model}'
        curr_model_path = f'../save/task2/{save_filename}'
        if os.path.exists(prev_model_path):
            self.model = keras.models.load_model(prev_model_path)
            print('Model anterior incarcat')

        else:
            self.model = self.build_model()

        #impartim datele in train si validation
        X_train, X_val, y_train, y_val = train_test_split(
            training_examples,
            train_labels,
            test_size=0.1,
            random_state=42,
            stratify=train_labels
        )

        self.model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"]
        )
        callbacks = [
            keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
            ),

            keras.callbacks.ModelCheckpoint(
            filepath="../save/task2/character_classifier_best.h5",
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1
            )
        ]

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=64,
            shuffle=True,
            callbacks=callbacks
        )

        self.model.save(curr_model_path)
        print('Model salvat.')
    
    def compute_average_precision(self, rec, prec):
        # functie adaptata din 2010 Pascal VOC development kit
        m_rec = np.concatenate(([0], rec, [1]))
        m_pre = np.concatenate(([0], prec, [0]))
        for i in range(len(m_pre) - 2, -1, -1):
            m_pre[i] = max(m_pre[i], m_pre[i + 1])
        m_rec = np.array(m_rec)
        i = np.where(m_rec[1:] != m_rec[:-1])[0] + 1
        average_precision = np.sum((m_rec[i] - m_rec[i - 1]) * m_pre[i])
        return average_precision

    def eval_detections(self, character_name, detections, scores, file_names):
        ground_truth_file = np.loadtxt(f'../validare/task2_{character_name}_gt_validare.txt', dtype='str')
        ground_truth_file_names = np.array(ground_truth_file[:, 0])
        ground_truth_detections = np.array(ground_truth_file[:, 1:], dtype=int)

        num_gt_detections = len(ground_truth_detections)  # numar total de adevarat pozitive
        gt_exists_detection = np.zeros(num_gt_detections)
        # sorteazam detectiile dupa scorul lor
        sorted_indices = np.argsort(scores)[::-1]
        file_names = file_names[sorted_indices]
        scores = scores[sorted_indices]
        detections = detections[sorted_indices]

        num_detections = len(detections)
        true_positive = np.zeros(num_detections)
        false_positive = np.zeros(num_detections)
        duplicated_detections = np.zeros(num_detections)

        for detection_idx in range(num_detections):
            indices_detections_on_image = np.where(ground_truth_file_names == file_names[detection_idx])[0]

            gt_detections_on_image = ground_truth_detections[indices_detections_on_image]
            bbox = detections[detection_idx]
            max_overlap = -1
            index_max_overlap_bbox = -1
            for gt_idx, gt_bbox in enumerate(gt_detections_on_image):
                overlap = intersection_over_union(bbox, gt_bbox)
                if overlap > max_overlap:
                    max_overlap = overlap
                    index_max_overlap_bbox = indices_detections_on_image[gt_idx]

            # clasifica o detectie ca fiind adevarat pozitiva / fals pozitiva
            if max_overlap >= 0.3:
                if gt_exists_detection[index_max_overlap_bbox] == 0:
                    true_positive[detection_idx] = 1
                    gt_exists_detection[index_max_overlap_bbox] = 1
                else:
                    false_positive[detection_idx] = 1
                    duplicated_detections[detection_idx] = 1
            else:
                false_positive[detection_idx] = 1

        cum_false_positive = np.cumsum(false_positive)
        cum_true_positive = np.cumsum(true_positive)

        rec = cum_true_positive / num_gt_detections
        prec = cum_true_positive / (cum_true_positive + cum_false_positive)
        average_precision = self.compute_average_precision(rec, prec)
        plt.plot(rec, prec, '-')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Average precision {character_name} %.3f' % average_precision)
        plt.savefig(os.path.join(self.params.dir_save_files_task2, f'precizie_medie_{character_name}.png'))
        plt.show()


    def classify(self, detections, file_names):

        # if self.model is None:
        #     X, y = self.build_dataset()
        #     self.train_classifier(X, y, 'classifier_first.h5', 'nomodel.h5')
        self.model = keras.models.load_model('../save/task2/character_classifier_best.h5')

        # construim un dictionar de tipul filename : [toate detectiile]
        dict = group_detections_by_filename(detections, file_names)
        detections_daphne = []
        detections_fred = []
        detections_velma = []
        detections_shaggy = []

        filenames_daphne = []
        filenames_fred = []
        filenames_velma = []
        filenames_shaggy = []

        scores_daphne = []
        scores_fred = []
        scores_velma = []
        scores_shaggy = []

        for filename in dict:
            print(f'Procesam imaginea {filename}')
            start_time = timeit.default_timer()
            detect = dict[filename]
            patches = []
            # incarcam imaginea 
            filename_path = os.path.join(self.params.dir_test_examples, filename)
            img = cv.imread(filename_path)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            # luam patchul corespunzator fiecarei detectii, resize la 64 x 64, add la patches
            for idx in range(len(detect)):
                xmin = detect[idx][0]
                ymin = detect[idx][1]
                xmax = detect[idx][2]
                ymax = detect[idx][3]
                patch = img[ymin:ymax, xmin:xmax]
                patch = cv.resize(patch, (64, 64))
                patch = patch.astype(np.float32) / 255.0
                patches.append(patch)
            # calculam predictiile modelului pe patches
            patches = np.array(patches, dtype=np.float32)
            preds = self.model.predict(patches, batch_size=64, verbose = 0)
            # preds are forma N x 5
            # N - nr de detectii din imagine
            # cate o probab pt fiecare clasa [0..4], prob insumate dau 1
            for idx in range(len(preds)):
                pred = preds[idx]
                bbox = detect[idx]
                pred_label = np.argmax(pred)
                if pred[pred_label] > 0.5: # daca are un confidence > 0.5
                    if pred_label == 0:
                        detections_daphne.append(bbox)
                        filenames_daphne.append(filename)
                        scores_daphne.append(pred[pred_label])
                    elif pred_label == 1:
                        detections_fred.append(bbox)
                        filenames_fred.append(filename)
                        scores_fred.append(pred[pred_label])
                    elif pred_label == 2:
                        detections_velma.append(bbox)
                        filenames_velma.append(filename)
                        scores_velma.append(pred[pred_label])
                    elif pred_label == 3:
                        detections_shaggy.append(bbox)
                        filenames_shaggy.append(filename)
                        scores_shaggy.append(pred[pred_label])
            end_time = timeit.default_timer()
            print(f'Timp procesare: {end_time - start_time:.2f} sec')
        # convertim la np arrays
        detections_daphne = np.array(detections_daphne, dtype=np.int32)
        detections_fred = np.array(detections_fred, dtype=np.int32)
        detections_velma = np.array(detections_velma, dtype=np.int32)
        detections_shaggy = np.array(detections_shaggy, dtype=np.int32)

        filenames_daphne = np.array(filenames_daphne)
        filenames_fred = np.array(filenames_fred)
        filenames_velma = np.array(filenames_velma)
        filenames_shaggy = np.array(filenames_shaggy)

        scores_daphne = np.array(scores_daphne, dtype=np.float32)
        scores_fred = np.array(scores_fred, dtype=np.float32)
        scores_velma = np.array(scores_velma, dtype=np.float32)
        scores_shaggy = np.array(scores_shaggy, dtype=np.float32)

        # salvam predictiile facute
        np.save('../save/task2/detections_daphne.npy', detections_daphne)
        np.save('../save/task2/detections_fred.npy', detections_fred)
        np.save('../save/task2/detections_velma.npy', detections_velma)
        np.save('../save/task2/detections_shaggy.npy', detections_shaggy)

        np.save('../save/task2/file_names_daphne.npy', filenames_daphne)
        np.save('../save/task2/file_names_fred.npy', filenames_fred)
        np.save('../save/task2/file_names_velma.npy', filenames_velma)
        np.save('../save/task2/file_names_shaggy.npy', filenames_shaggy)

        np.save('../save/task2/scores_daphne.npy', scores_daphne)
        np.save('../save/task2/scores_fred.npy', scores_fred)
        np.save('../save/task2/scores_velma.npy', scores_velma)
        np.save('../save/task2/scores_shaggy.npy', scores_shaggy)

        # evaluam detectiile  scoatem astea
        # self.eval_detections('daphne', detections_daphne, scores_daphne, filenames_daphne)
        # self.eval_detections('fred', detections_fred, scores_fred, filenames_fred)
        # self.eval_detections('velma', detections_velma, scores_velma, filenames_velma)
        # self.eval_detections('shaggy', detections_shaggy, scores_shaggy, filenames_shaggy)



