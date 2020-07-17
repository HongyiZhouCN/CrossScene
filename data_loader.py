import tensorflow as tf
import os
import numpy as np

def preprocess_image(image):
  image = tf.image.decode_jpeg(image)
  image = tf.image.resize(image, [72,72])
  image /= 255.0  # normalize to [0,1] range
  return image

def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)


class ImgLoader:

    def __init__(self,Path_img,Path_den, batch_size):
        self.Path_img = Path_img
        self.Path_den = Path_den
        self.files_img = os.listdir(Path_img)
        self.files_den = os.listdir(Path_den)
        self.ImgList = []
        self.DenList = []
        self.batch_size = batch_size

    def data_generator_density(self): #data loader for count training
        all_index = list(range(0, len(self.files_img)))
        start_index = 0
        while True:
            if start_index + self.batch_size >= len(all_index):
                np.random.shuffle(all_index)
                continue
            batch_input_image, batch_output_map, dumm_zero_batch = [], [], []
            for index in range(start_index, start_index+ self.batch_size):
                input_img = load_and_preprocess_image(os.path.join(self.Path_img,self.files_img[all_index[index]]))
                output_den = np.genfromtxt(os.path.join(self.Path_den,self.files_den[all_index[index]]), delimiter=',')
                batch_input_image.append(input_img)
                batch_output_map.append(output_den)
                dumm_zero_batch.append(0.)
            start_index += self.batch_size
            yield np.array(batch_input_image), [np.array(dumm_zero_batch), np.array(batch_output_map)]

    def data_generator_count(self): # data loader for density_map training
        all_index = list(range(0, len(self.files_img)))
        start_index = 0
        while True:
            if start_index + self.batch_size >= len(all_index):
                np.random.shuffle(all_index)
                continue
            batch_input_image, batch_density_count, batch_output_count = [], [], []
            for index in range (start_index, start_index + self.batch_size):
                input_img = load_and_preprocess_image(os.path.join(self.Path_img, self.files_img[all_index[index]]))
                output_den =np.genfromtxt(os.path.join(self.Path_den, self.files_den[all_index[index]]), delimiter=',')
                #calculate the output count from density map
                output_count = 0
                for i in range(0,18):
                    for j in range(0,18):
                        output_count += output_den[i][j]
                batch_input_image.append(input_img)
                batch_density_count.append(output_den)
                batch_output_count.append(output_count)

            start_index += self.batch_size
            #loader for model_1
            #yield (np.array(batch_input_image), {'count_output': np.array(batch_output_count), 'density_output' : np.array(batch_density_count)})
            yield np.array(batch_input_image), np.array(batch_output_count)


    def data_generator_both(self): # data loader for density_map training
        all_index = list(range(0, len(self.files_img)))
        start_index = 0
        while True:
            if start_index + self.batch_size >= len(all_index):
                np.random.shuffle(all_index)
                continue
            batch_input_image, batch_density_count, batch_output_count = [], [], []
            for index in range (start_index, start_index + self.batch_size):
                input_img = load_and_preprocess_image(os.path.join(self.Path_img, self.files_img[all_index[index]]))
                output_den =np.genfromtxt(os.path.join(self.Path_den, self.files_den[all_index[index]]), delimiter=',')
                #calculate the output count from density map
                output_count = 0
                for i in range(0,18):
                    for j in range(0,18):
                        output_count += output_den[i][j]
                batch_input_image.append(input_img)
                batch_density_count.append(output_den)
                batch_output_count.append(output_count)

            start_index += self.batch_size
            #loader for model_1
            #yield (np.array(batch_input_image), {'count_output': np.array(batch_output_count), 'density_output' : np.array(batch_density_count)})
            yield np.array(batch_input_image), [np.array(batch_output_count), np.array(batch_density_count)]




