'''
Implementation of Eigenfaces face recognition using PCA.

All images must be the same size and converted to grayscale.
Each person/class must have his/her own folder, and every photo of a given
person must be placed into the same folder, and the folder must be given the
person/class's name.

The algorithm is run from the main.py module.
When run, the algorithm will print the predicted classification to the console
and will show the picture.
'''

from PIL import Image

import numpy as np
#import pylab
import sys
import glob
import os

from . import pca


class EigenFaces(object):
    def train(self, root_training_images_folder):
        self.projected_classes = []

        self.list_of_arrays_of_images, self.labels_list, \
            list_of_matrices_of_flattened_class_samples = \
                read_images(root_training_images_folder)

         # create matrix to store all flattened images
        images_matrix = np.array([np.array(Image.fromarray(img)).flatten()
              for img in self.list_of_arrays_of_images],'f')

        # perform PCA
        self.eigenfaces_matrix, variance, self.mean_image = pca.pca(images_matrix)

        # Projecting each class sample (as class matrix) and then using the class average as the class weights for comparison with the Target image
        for class_sample in list_of_matrices_of_flattened_class_samples:
            class_weights_vertex = self.project_image(class_sample)
            self.projected_classes.append(class_weights_vertex.mean(0))

    def project_image(self, X):
        X = X - self.mean_image
        return np.dot(X, self.eigenfaces_matrix.T)

    def predict_face(self, X):
        min_class = -1
        min_distance = np.finfo('float').max
        projected_target = self.project_image(X)
        # delete last array item, it's nan
        projected_target = np.delete(projected_target, -1)
        for i in range(len(self.projected_classes)):
            distance = np.linalg.norm(projected_target - np.delete(self.projected_classes[i], -1))
            if distance < min_distance:
                min_distance = distance
                min_class = self.labels_list[i]
        # print(min_class, min_distance)
        return min_class

    def __repr__(self):
        return "PCA (num_components=%d)" % (self._num_components)


def read_images(path, sz=None):
    """Reads the images in a given folder, resizes images on the fly if size is given.

    Args:
        path: Path to a folder with subfolders representing the subjects (persons).
        sz: A tuple with the size Resizes

    Returns:
        A tuple of (images, image_labels, class_matrix_list) where
            images: The images, which is a Python list of numpy arrays.
            image_labels: The corresponding labels (the unique number of
            the subject, person).
    """
    class_samples_list = []
    class_matrices_list = []
    images, image_labels = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            class_samples_list = []
            for filename in os.listdir(subject_path):
                if filename != ".DS_Store":
                    try:
                        im = Image.open(os.path.join(subject_path, filename))
                        # resize to given size (if given) e.g., sz = (480, 640)
                        if (sz is not None):
                            im = im.resize(sz, Image.ANTIALIAS)
                        images.append(np.asarray(im, dtype = np.uint8))

                    except IOError as e:
                        errno, strerror = e.args
                        print("I/O error({0}): {1}".format(errno, strerror))
                    except:
                        print("Unexpected error:", sys.exc_info()[0])
                        raise
                    # adds each sample within a class to this List
                    class_samples_list.append(np.asarray(im, dtype = np.uint8))

            # flattens each sample within a class and adds the array/vector to a class matrix
            class_samples_matrix = np.array([img.flatten()
                for img in class_samples_list],'f')

             # adds each class matrix to this MASTER List
            class_matrices_list.append(class_samples_matrix)

            image_labels.append(subdirname)

    return images, image_labels, class_matrices_list
