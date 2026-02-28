from Parameters import *
from Utils import *
import numpy as np
import keras.models
import matplotlib.pyplot as plt
import glob
import cv2 as cv
import ntpath
import timeit
from keras.layers import (
    Conv2D, MaxPooling2D,
    Dense, Dropout,
    GlobalAveragePooling2D
)
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from pathlib import Path


class FacialDetector:
    def __init__(self, params:Parameters):
        self.params = params
        self.model = None

# ------------ extragere imagini  ---------------------------
    def extract_pos_examples(self, base_directory='../antrenare'):
        output_dir = '../examples/positiveExamples'
        log_dir = '../examples'
        log_file = os.path.join(log_dir, 'log_extraction.txt')
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(log_file, 'w', encoding='utf-8') as log:
            dirs = [d.name for d in Path(base_directory).iterdir() if d.is_dir()]
            for directory in dirs:
                filename = f'{directory}_annotations.txt'
                filename_path = os.path.join(base_directory, filename)
                directory_path = os.path.join(base_directory, directory)
                with open(filename_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip()
                        line = line.split()
                        image_name = line[0]
                        xmin, ymin, xmax, ymax = int(line[1]), int(line[2]), int(line[3]), int(line[4])
                        character_name = line[5]
                        # citim imaginea coresp
                        img_path = os.path.join(directory_path, image_name)
                        img = cv.imread(img_path)

                        # verificam daca in output dirs exista directorul cu numele caracterului identificat
                        character_directory = os.path.join(output_dir, character_name)
                        if not os.path.exists(character_directory):
                            os.makedirs(character_directory)
                        cropped_image = img[ymin:ymax, xmin:xmax]
                        
                        # verificam daca imaginea cropata are dimensiuni valide
                        cropped_image_resized = cv.resize(cropped_image, (self.params.dim_window, self.params.dim_window))
                        num_files = len(os.listdir(character_directory))
                        output_filename = f'{num_files + 1}.jpg'
                        output_path = os.path.join(character_directory, output_filename)
                        cv.imwrite(output_path, cropped_image_resized)
                        msg = f'Imaginea {output_filename} din {character_name} este generata de {img_path}'
                        log.write(msg + '\n')
            print(f'\nLog salvat in: {log_file}')


    def get_positive_images(self):
        # incarcam imaginile pozitive
        # output: N x D, N - nr de img poz
        # D - (64, 64, 3), valorile pixelilor

        # verificam daca nu exista directorul cu example pozitive
        if not os.path.exists(self.params.dir_pos_examples):
            self.extract_pos_examples()

        positive_images = []
        dirs = [d.name for d in Path(self.params.dir_pos_examples).iterdir() if d.is_dir()]

        for directory in dirs:
            curr_directory = os.path.join(self.params.dir_pos_examples, directory)
            images_path = os.path.join(curr_directory, '*.jpg')
            files = glob.glob(images_path)
            num_images = len(files)
            print(f'In folderul {directory} sunt {num_images} imagini pozitive')
            for i in range(num_images):
                img = cv.imread(files[i]) # BGR image
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0 # normalizam valorile pixelilor
                positive_images.append(img)
                if self.params.use_flip_images:
                    positive_images.append(np.fliplr(img))
                if self.params.augument_positives:
                    positive_images.append(aug_blur(img))
                    positive_images.append(aug_rotate(img, 5))
        positive_images = np.array(positive_images)
        return positive_images

    def extract_neg_examples(self, base_directory='../antrenare', min_patch_size = 64, max_patch_size = 90, neg_per_image = 20, max_iou = 0.1):
        output_dir = '../examples/negativeExamples'
        no_examples = 0
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        dirs = [d.name for d in Path(base_directory).iterdir() if d.is_dir()]
        for directory in dirs:
                # luam pe rand fiecare subfolder din antrenare
                filename = f'{directory}_annotations.txt'
                filename_path = os.path.join(base_directory, filename)
                directory_path = os.path.join(base_directory, directory)
                with open(filename_path, 'r') as f:
                    lines = f.readlines()
                #parcurgem imaginile din directorul respectiv
                images_path = os.path.join(directory_path, '*.jpg')
                files = glob.glob(images_path)
                files_number = len(files)
                for i in range(files_number):
                    print(f'A inceput procesarea imaginii {i}/{files_number} din folderul {directory}')
                    img = cv.imread(files[i]) # 480 x 360
                    img_name = os.path.basename(files[i])
                    num_rows = img.shape[0]
                    num_cols = img.shape[1]
                    annotations = [line.strip().split() for line in lines if line.strip().split()[0] == img_name]
                    xmin = [int(annot[1]) for annot in annotations]
                    ymin = [int(annot[2]) for annot in annotations]
                    xmax = [int(annot[3]) for annot in annotations]
                    ymax = [int(annot[4]) for annot in annotations]
                    ct = neg_per_image
                    while ct > 0: # mai trebuie sa alegem patchuri
                        # selectam random o dimensiune pentru patch
                        patch_w = np.random.randint(min_patch_size, max_patch_size+1)
                        patch_l = np.random.randint(min_patch_size, max_patch_size+1)
                        xmin_patch = np.random.randint(0, num_cols - patch_l)
                        ymin_patch = np.random.randint(0, num_rows - patch_w)
                        xmax_patch = xmin_patch + patch_l
                        ymax_patch = ymin_patch + patch_w
                        patch_coords = [xmin_patch, ymin_patch, xmax_patch, ymax_patch]
                        is_face = False
                        # verificam iou cu toate detectiile din imaginea resp
                        # nu vrem sa contaminam exemplele negative cu fete, avem un 
                        # max_iou permis
                        for idx in range(len(xmin)):
                            y_start = max(0, ymin[idx])
                            y_end = min(num_rows, ymax[idx])
                            x_start = max(0, xmin[idx])
                            x_end = min(num_cols, xmax[idx])
                            annot_coords = [x_start, y_start, x_end, y_end]
                            iou = intersection_over_union(patch_coords, annot_coords)
                            if iou > max_iou:
                                is_face = True
                        if not is_face:
                            no_examples += 1
                            output_filename = f'{no_examples}.jpg'
                            output_path = os.path.join(output_dir, output_filename)
                            patch = img[ymin_patch:ymax_patch, xmin_patch:xmax_patch]
                            resized_patch = cv.resize(patch, 
                                                      (self.params.dim_window, 
                                                       self.params.dim_window)
                                                    )
                            cv.imwrite(output_path, resized_patch)
                            ct -= 1
                    

    def get_negative_images(self):
        # incarcam imaginile negative
        # output: N x D, N - nr de img neg
        # D - (64, 64, 3), valorile pixelilor

        if not os.path.exists(self.params.dir_neg_examples):
            self.extract_neg_examples()

        images_path = os.path.join(self.params.dir_neg_examples, '*.jpg')
        files = glob.glob(images_path)
        num_images = len(files)
        print (f'Numarul de imagini negative este {num_images}')
        negative_images = []
        print('Calculam descriptorii pt %d imagini negative' % num_images)
        for i in range(num_images):
            img = cv.imread(files[i]) # BGR image
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0 # normalizam valorile pixelilor
            negative_images.append(img)
        negative_images = np.array(negative_images)
        return negative_images
    
    # folosim un model antrenat pe exemple pozitive (fetele personajelor)
    # si exemple negative (patchuri random extrase din imagini)
    # parcurgem datele de antrenare si luam acele patchuri pentru care
    # modelul curent ofera un scor > min_score si care au un max_iou cu 
    # fetele din imaginea respectiva
    # vrem sa reducem numarul de false positives ale modelului
    def mine_hard_negatives(self, images_dir, gt_txt_path, out_dir = '../examples/hardNegatives',
                        min_score=0.8, iou_neg=0.05, iou_partial=0.1,
                        max_per_image=10, scales=[1.3, 1.2, 1.1,1.0,0.9,0.8,0.7,0.6,0.5,0.4], 
                        stride=8, batch_size=512,
                        min_center_dist=20.0, sel_iou_thresh=0.30):

        os.makedirs(out_dir, exist_ok=True)
        # construim un dictionar de tipul
        # nume_imagine : [lista detectiilor din imagine]
        gt = np.loadtxt(gt_txt_path, dtype=str, ndmin=2) 
        gt_dict = {}
        for row in gt:
            fname = row[0]
            x1, y1, x2, y2 = map(int, row[1:5])
            gt_dict.setdefault(fname, []).append([x1, y1, x2, y2])
 
        if self.model is None:
            model_path = os.path.join(self.params.dir_save_files_task1, 'model_first.h5')
            self.model = keras.models.load_model(model_path)

        dim = self.params.dim_window
        img_paths = glob.glob(os.path.join(images_dir, "*.jpg"))
        # salvam in acelasi director false negatives de la mai multe personaje
        # nu vrem sa le suprascriem
        img_already_saved = glob.glob(os.path.join(out_dir, "*.jpg"))
        saved = 0
        if img_already_saved is not None:
            saved = len(img_already_saved) + 1

        for img_path in img_paths: # luam fiecare imagine
            fname = ntpath.basename(img_path)
            if fname not in gt_dict: # daca nu exista detectii pentru img respectiva
                continue

            img_bgr = cv.imread(img_path)
            img = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
            H, W = img.shape[0], img.shape[1]
            detections = gt_dict[fname]
            collected = 0

            # nu vrem sa avem false positives foarte apropiate, vrem 
            # sa surprindem FPs cat mai diverse, deci evitam imaginile prea similare
            selected_boxes = []      # boxuri acceptate in coordonate originale
            selected_centers = []    # centrele boxurilor acceptate

            for sc in scales:
                interp = cv.INTER_AREA if sc < 1.0 else cv.INTER_LINEAR
                scaled = cv.resize(img, (0, 0), fx=sc, fy=sc, interpolation=interp)
                sh, sw = scaled.shape[0], scaled.shape[1]
                # imaginea redimensionata sa nu fie < fereastra glisanta
                if sh < dim or sw < dim:
                    continue

                patches = []
                coords = []
                # trecem fereastra glisanta peste img redim
                for y in range(0, sh - dim + 1, stride):
                    for x in range(0, sw - dim + 1, stride):
                        patches.append(scaled[y:y+dim, x:x+dim])
                        coords.append((x, y))

                patches = np.asarray(patches, dtype=np.float32)
                # modelul asociaza fiecarui patch un scor din [0, 1]
                # 0 - non fata, 1 - fata
                preds = self.model.predict(patches, batch_size=batch_size, verbose=0).ravel()

                # sortam descrescator, vrem cele mai puternice FPs 
                order = np.argsort(preds)[::-1]

                for idx in order:
                    score = float(preds[idx])
                    if score < min_score:
                        break

                    x, y = coords[idx]
                    # revenim la coordonatele originale
                    x1 = int(x / sc)
                    y1 = int(y / sc)
                    x2 = int((x + dim) / sc)
                    y2 = int((y + dim) / sc)

                    # clamp
                    x1 = max(0, min(x1, W - 1))
                    y1 = max(0, min(y1, H - 1))
                    x2 = max(0, min(x2, W))
                    y2 = max(0, min(y2, H))
                    if x2 <= x1 or y2 <= y1:
                        continue

                    # calculam max_iou cu toate detectiile gt din imaginea respectiva
                    max_iou = 0.0
                    for gt_box in detections:
                        max_iou = max(max_iou, intersection_over_union([x1, y1, x2, y2], gt_box))

                    # daca se intersecteaza prea mult cu o detectie, nu il luam in calcul
                    if not (max_iou < iou_neg or (iou_neg <= max_iou <= iou_partial)):
                        continue
                    
                    # calculam coords centrului
                    cx = 0.5 * (x1 + x2)
                    cy = 0.5 * (y1 + y2)

                    # verificam sa nu fie prea apropiat ca centru cu celelalte 
                    # FPs selectate din aceasta imagine
                    too_close = False
                    for pcx, pcy in selected_centers:
                        dx = cx - pcx
                        dy = cy - pcy
                        if (dx * dx + dy * dy) ** 0.5 < min_center_dist:
                            too_close = True
                            break
                    if too_close:
                        continue

                    # calculam max_iou cu celelalte FPs selecte
                    # din imagine
                    max_iou_sel = 0.0
                    for b in selected_boxes:
                        max_iou_sel = max(max_iou_sel, intersection_over_union([x1, y1, x2, y2], b))
                    if max_iou_sel > sel_iou_thresh:
                        continue

                    # acceptam si salvam coords
                    selected_boxes.append([x1, y1, x2, y2])
                    selected_centers.append((cx, cy))

                    # am normalizat imaginea pentru a putea rula modelul pe ea
                    # o denormalizam pentru a o putea salva
                    patch_u8 = (img[y1:y2, x1:x2] * 255.0).astype(np.uint8)
                    patch_u8 = cv.resize(patch_u8, (dim, dim), interpolation=cv.INTER_LINEAR)

                    out_path = os.path.join(out_dir, f"hn_{saved:07d}.jpg")
                    cv.imwrite(out_path, cv.cvtColor(patch_u8, cv.COLOR_RGB2BGR))
                    saved += 1
                    collected += 1

                    if collected >= max_per_image:
                        break
                if collected >= max_per_image:
                        break

        print(f"S-au salvat hard negatives din {images_dir} in {out_dir}")


    def get_negative_images_hm(self, path):
        characters = ['daphne', 'fred', 'velma', 'shaggy']
        # pt fiecare caracter generam hard negatives daca nu 
        # sunt deja generate
        if not os.path.exists(path):
            for ch in characters:
                dir = f'../antrenare/{ch}'
                gt = f'../antrenare/{ch}_annotations.txt'
                self.mine_hard_negatives(
                images_dir=dir,        
                gt_txt_path=gt,
                )
        images_path = os.path.join(path, '*.jpg')
        files = glob.glob(images_path)
        num_images = len(files)
        print (f'Numarul de imagini hard negative este {num_images}')
        negative_images = []
        for i in range(num_images):
            img = cv.imread(files[i]) # BGR image
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0 # normalizam valorile pixelilor
            negative_images.append(img)
        negative_images = np.array(negative_images)
        return negative_images

# --------------- model -----------------------
    def build_model(self):
        model = Sequential([
        Conv2D(32, 3, activation='relu', padding='same', input_shape=(64,64,3)),
        MaxPooling2D(),

        Conv2D(64, 3, activation='relu', padding='same'),
        MaxPooling2D(),

        Conv2D(128, 3, activation='relu', padding='same'),
        GlobalAveragePooling2D(),

        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
        return model


    def train_classifier(self, training_examples, train_labels, save_filename, prev_model):

        prev_model_path = os.path.join(self.params.dir_save_files_task1, prev_model)
        curr_model_path = os.path.join(self.params.dir_save_files_task1, save_filename)
        # exista un model existent, continuam sa il antrenam pe acela
        if os.path.exists(prev_model_path):
            self.model = keras.models.load_model(prev_model_path)
            print('Model anterior incarcat.')

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

        # sunt mai multe exemple neg decat pozitive
        # vrem o penalizare mai mare pentru gresirea exemplelor poz
        class_weight = {
            0: 1.0,
            1: 2.0
        }

        print('Class weights:', class_weight)

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=4,
                restore_best_weights=True
            )
        ]

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=20,
            batch_size=64,
            shuffle=True,
            class_weight=class_weight,
            callbacks=callbacks
        )



        self.model.save(curr_model_path)
        print('Model salvat.')

        scores = self.model.predict(X_train, verbose=0).ravel()

        positive_scores = scores[y_train == 1]
        negative_scores = scores[y_train == 0]

        plt.figure(figsize=(10, 4))
        plt.plot(np.sort(positive_scores), label='Pozitive', linewidth=2)
        plt.plot(np.sort(negative_scores), label='Negative', linewidth=2)
        plt.axhline(self.params.threshold, color='k', linestyle='--', label=f'Threshold = {self.params.threshold}')
        plt.xlabel('Exemple')
        plt.ylabel('Scor sigmoid')
        plt.title('Distributia scorurilor dupa antrenare')
        plt.legend()
        plt.grid(True)
        plt.show()




    def non_maximal_suppression(self, image_detections, image_scores, image_size):
        """
        Detectiile cu scor mare suprima detectiile ce se suprapun cu acestea dar au scor mai mic.
        Detectiile se pot suprapune partial, dar centrul unei detectii nu poate
        fi in interiorul celeilalte detectii.
        :param image_detections:  numpy array de dimensiune NX4, unde N este numarul de detectii.
        :param image_scores: numpy array de dimensiune N
        :param image_size: tuplu, dimensiunea imaginii
        :return: image_detections si image_scores care sunt maximale.
        """

        # xmin, ymin, xmax, ymax
        x_out_of_bounds = np.where(image_detections[:, 2] > image_size[1])[0]
        y_out_of_bounds = np.where(image_detections[:, 3] > image_size[0])[0]
        #print(x_out_of_bounds, y_out_of_bounds)
        image_detections[x_out_of_bounds, 2] = image_size[1]
        image_detections[y_out_of_bounds, 3] = image_size[0]
        sorted_indices = np.flipud(np.argsort(image_scores))
        sorted_image_detections = image_detections[sorted_indices]
        sorted_scores = image_scores[sorted_indices]

        is_maximal = np.ones(len(image_detections)).astype(bool)
        iou_threshold = 0.3 #0.3
        for i in range(len(sorted_image_detections) - 1):
            if is_maximal[i] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                for j in range(i + 1, len(sorted_image_detections)):
                    if is_maximal[j] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                        if intersection_over_union(sorted_image_detections[i],sorted_image_detections[j]) > iou_threshold:is_maximal[j] = False
                        else:  # verificam daca centrul detectiei este in mijlocul detectiei cu scor mai mare
                            c_x = (sorted_image_detections[j][0] + sorted_image_detections[j][2]) / 2
                            c_y = (sorted_image_detections[j][1] + sorted_image_detections[j][3]) / 2
                            if sorted_image_detections[i][0] <= c_x <= sorted_image_detections[i][2] and \
                                    sorted_image_detections[i][1] <= c_y <= sorted_image_detections[i][3]:
                                is_maximal[j] = False
        return sorted_image_detections[is_maximal], sorted_scores[is_maximal]

    def run(self):

        if self.model is None:
            model_path = os.path.join(self.params.dir_save_files_task1, 'model_post_hm.h5')
            self.model = keras.models.load_model(model_path)

        threshold = self.params.threshold

        test_files = glob.glob(os.path.join(self.params.dir_test_examples, '*.jpg'))

        detections, scores, file_names = [], [], []
        ct = 0
        num_test_images = len(test_files)
        for img_path in test_files:
            start_time = timeit.default_timer()
            ct += 1
            img = cv.imread(img_path)
            if img is None: 
                continue
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            print(f'Procesam imaginea {ct}/{num_test_images}')
            image_detections = []
            image_scores = []
            # exista fete foarte mici si avem nevoie de un scale 2 pentru a 
            # avea o fereastra destul de fit pentru a fi detectate fetele
            scales = [2, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]

            for scale in scales:
                interp = cv.INTER_LINEAR if scale >= 1.0 else cv.INTER_AREA
                scaled = cv.resize(img, None, fx=scale, fy=scale, interpolation=interp)

                if scaled.shape[0] < self.params.dim_window or scaled.shape[1] < self.params.dim_window:
                    continue

                patches = []
                coords = []

                # self.params.step mentinem doar pentru scalarile > 0.5
                step = self.params.step
                if scale < 0.6:
                    step = 4

                for y in range(0, scaled.shape[0] - self.params.dim_window + 1, step):
                    for x in range(0, scaled.shape[1] - self.params.dim_window + 1, step):
                        patch = scaled[y:y+self.params.dim_window, x:x+self.params.dim_window]
                        patches.append(patch)
                        coords.append((x, y, scale))

                patches = np.array(patches, dtype=np.float32)
                preds = self.model.predict(patches, batch_size=512, verbose = 0).ravel()

                for (x, y, scale), score in zip(coords, preds):
                    if score < threshold:
                        continue

                    x1 = int(x / scale)
                    y1 = int(y / scale)
                    x2 = int((x + self.params.dim_window) / scale)
                    y2 = int((y + self.params.dim_window) / scale)

                    image_detections.append([x1, y1, x2, y2])
                    image_scores.append(score)
            # nms pe toate scale urile
            if image_scores:
                image_detections, image_scores = self.non_maximal_suppression(
                    np.array(image_detections),
                    np.array(image_scores),
                    img.shape
                )

                detections.extend(image_detections)
                scores.extend(image_scores)
                file_names.extend([ntpath.basename(img_path)] * len(image_scores))

            end_time = timeit.default_timer()
            print(f'Timp procesare: {end_time - start_time:.2f} sec')
        return np.array(detections), np.array(scores), np.array(file_names)


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

    def eval_detections(self, gt_file_name, detections, scores, file_names):
        ground_truth_file = np.loadtxt(gt_file_name, dtype='str')
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
        plt.title('Average precision %.3f' % average_precision)
        save_path = os.path.join(self.params.dir_save_files_task1, 'precizie_medie.png')
        plt.savefig(save_path)
        plt.show()
