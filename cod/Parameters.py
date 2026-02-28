import os

class Parameters:
    def __init__(self):
        self.base_dir = '../examples'
        self.dir_pos_examples = os.path.join(self.base_dir, 'positiveExamples')
        self.dir_neg_examples = os.path.join(self.base_dir, 'negativeExamples')
        self.dir_test_examples = '../testare' # modificat aici
        self.path_annotations = '../validare/task1_gt_validare.txt'
        self.dir_save_files_task1 = '../save/task1'
        self.dir_save_files_task2 = '../save/task2'
        if not os.path.exists(self.dir_save_files_task1):
            os.makedirs(self.dir_save_files_task1)
            print('directory created: {} '.format(self.dir_save_files_task1))
        else:
            print('directory {} exists '.format(self.dir_save_files_task1))

        if not os.path.exists(self.dir_save_files_task2):
            os.makedirs(self.dir_save_files_task2)
            print('directory created: {} '.format(self.dir_save_files_task2))
        else:
            print('directory {} exists '.format(self.dir_save_files_task2))

        # set the parameters
        self.dim_window = 64  # exemplele pozitive (fete de oameni cropate) au 64x64 pixeli
        self.overlap = 0.3
        self.number_positive_examples = 6547  # numarul exemplelor pozitive
        self.number_negative_examples = 80000# numarul exemplelor negative
        self.has_annotations = True
        self.threshold = 0.8
        self.step = 8
        self.augument_positives = True
