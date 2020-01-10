from Camera import Camera
from SingleImage import SingleImage
from MatrixMethods import Compute3DRotationMatrix, Compute3DRotationDerivativeMatrix, ComputeSkewMatrixFromVector
import numpy as np
import numpy.linalg as la


class ImagePair(object):

    def __init__(self, image1, image2):
        """
        Initialize the ImagePair class
        :param image1: First image
        :param image2: Second image
        """
        self.__image1 = image1
        self.__image2 = image2
        self.__relativeOrientationImage1 = np.array([0, 0, 0, 0, 0, 0])  # The relative orientation of the first image
        self.__relativeOrientationImage2 = None  # The relative orientation of the second image
        self.__absoluteOrientation = None
        self.__isSolved = False  # Flag for the relative orientation

    @property
    def isSolved(self):
        """
        Flag for the relative orientation
        returns True if the relative orientation is solved, otherwise it returns False

        :return: boolean, True or False values
        """
        return self.__isSolved

    @property
    def RotationMatrix_Image1(self):
        """
        return the rotation matrix of the first image

        :return: rotation matrix

        :rtype: np.array 3x3
        """
        return Compute3DRotationMatrix(self.__relativeOrientationImage1[0], self.__relativeOrientationImage1[1],
                                       self.__relativeOrientationImage1[2])

    @property
    def RotationMatrix_Image2(self):
        """
        return the rotation matrix of the second image

        :return: rotation matrix

        :rtype: np.array 3x3
        """
        return Compute3DRotationMatrix(self.__relativeOrientationImage2[0], self.__relativeOrientationImage2[1],
                                       self.__relativeOrientationImage2[2])

    @property
    def PerspectiveCenter_Image1(self):
        """
        return the perspective center of the first image

        :return: perspective center

        :rtype: np.array (3, )
        """
        return self.__relativeOrientationImage1[0:3]

    @property
    def PerspectiveCenter_Image2(self):
        """
        return the perspective center of the second image

        :return: perspective center

        :rtype: np.array (3, )
        """
        return self.__relativeOrientationImage2[0:3]

    def ImagesToGround(self, imagePoints1, imagePoints2, Method=None):
        """
        Computes ground coordinates of homological points

        :param imagePoints1: points in image 1
        :param imagePoints2: corresponding points in image 2
        :param Method: method to use for the ray intersection, three options exist: geometric, vector, Collinearity

        :type imagePoints1: np.array nx2
        :type imagePoints2: np.array nx2
        :type Method: string

        :return: ground points, their accuracies.

        :rtype: dict

        .. warning::

            This function is empty, need implementation


        **Usage example**

        .. code-block:: py

            camera = Camera(152, None, None, None, None)
            image1 = SingleImage(camera)
            image2 = SingleImage(camera)

            imagePoints1 = np.array([[-4.83,7.80],
                                [-4.64, 134.86],
                                [5.39,-100.80],
                                [4.58,55.13],
                                [98.73,9.59],
                                [62.39,128.00],
                                [67.90,143.92],
                                    [56.54,-85.76]])
            imagePoints2 = np.array([[-83.17,6.53],
                                 [-102.32,146.36],
                                 [-62.84,-102.87],
                                 [-97.33,56.40],
                                 [-3.51,14.86],
                                 [-27.44,136.08],
                                 [-23.70,152.90],
                                 [-8.08,-78.07]])

            new = ImagePair(image1, image2)

            new.ImagesToGround(imagePoints1, imagePoints2, 'geometric'))

        """
        picpoints_3574_mm = self.__image1.ImageToCamera(imagePoints1)
        picpoints_3575_mm = self.__image1.ImageToCamera(imagePoints2)
        exori_XYZ_1 = self.__image1.exteriorOrientationParameters[0:3]
        exori_XYZ_2 = self.__image2.exteriorOrientationParameters[0:3]

        result_Gpoints = []
        dist_e = []

        for i in range(picpoints_3574_mm.shape[0]): #calculating per point set
            # following the geometric method for forward intersection:
            x_img1 = np.hstack((picpoints_3574_mm[i, :], -self.__image1.camera.focalLength)) / 1000  # to meter
            x_img2 = np.hstack((picpoints_3575_mm[i, :], -self.__image2.camera.focalLength)) / 1000
            v_img1 = (np.dot(Compute3DRotationMatrix(self.__image1.exteriorOrientationParameters[3],\
                                                     self.__image1.exteriorOrientationParameters[4],\
                                                     self.__image1.exteriorOrientationParameters[5]), x_img1)).reshape(3, 1)  # Rotating vector +T
            v_img2 = (np.dot(Compute3DRotationMatrix(self.__image1.exteriorOrientationParameters[3],\
                                                     self.__image1.exteriorOrientationParameters[4],\
                                                     self.__image1.exteriorOrientationParameters[5]), x_img2)).reshape(3, 1)  # Rotating vector +T
            v_img1 /= la.norm(v_img1)  # normalization
            v_img2 /= la.norm(v_img2)

            # Creating proper vectors
            vvt_img1 = np.dot(v_img1, v_img1.T)
            vvt_img2 = np.dot(v_img2, v_img2.T)
            I = np.eye(v_img1.shape[0])

            # Partial derivatives
            A_img1 = I - v_img1
            A_img2 = I - v_img2

            # L vector
            l1 = np.dot(A_img1, exori_XYZ_1)
            l2 = np.dot(A_img2, exori_XYZ_2)

            # Stack
            A = np.vstack((A_img1, A_img2))
            l = np.hstack((l1, l2))

            # Direct solution (no iterations needed)
            X = np.dot(la.inv(np.dot(A.T, A)), np.dot(A.T, l))
            # dist_e1 = np.dot((I - vvt_img1), X - exori_XYZ_1)
            dist_e1 = np.dot(A_img1, X)- l1
            # dist_e2 = np.dot((I - vvt_img2), X - exori_XYZ_2)
            dist_e2 = np.dot(A_img2, X)- l2

            dist_e.append((np.abs(dist_e1) + np.abs(dist_e2)) / 2) #Average
            result_Gpoints.append(X)

        return np.array(result_Gpoints), np.array(dist_e)

    def ComputeDependentRelativeOrientation(self, imagePoints1, imagePoints2, initialValues):
        """
         Compute relative orientation parameters

        :param imagePoints1: points in the first image [m"m]
        :param imagePoints2: corresponding points in image 2(homology points) nx2 [m"m]
        :param initialValues: approximate values of relative orientation parameters

        :type imagePoints1: np.array nx2
        :type imagePoints2: np.array nx2
        :type initialValues: np.array (6L,)

        :return: relative orientation parameters.

        :rtype: np.array 5x1 / ADD

        .. warning::

            Can be held either as dictionary or array. For your implementation and decision.

        .. note::

            Do not forget to decide how it is held and document your decision


        **Usage example**

        .. code-block:: py

            camera = Camera(152, None, None, None, None)
            image1 = SingleImage(camera)
            image2 = SingleImage(camera)

            imagePoints1 = np.array([[-4.83,7.80],
                                [-4.64, 134.86],
                                [5.39,-100.80],
                                [4.58,55.13],
                                [98.73,9.59],
                                [62.39,128.00],
                                [67.90,143.92],
                                [56.54,-85.76]])
            imagePoints2 = np.array([[-83.17,6.53],
                                 [-102.32,146.36],
                                 [-62.84,-102.87],
                                 [-97.33,56.40],
                                 [-3.51,14.86],
                                 [-27.44,136.08],
                                 [-23.70,152.90],
                                 [-8.08,-78.07]])
            new = ImagePair(image1, image2)

            new.ComputeDependentRelativeOrientation(imagePoints1, imagePoints2, np.array([1, 0, 0, 0, 0, 0])))

        """
        pass  # delete after implementation

    def Build_A_B_W(self, cameraPoints1, cameraPoints2, x):
        """
        Function for computing the A and B matrices and vector w.
        :param cameraPoints1: points in the first camera system
        :param ImagePoints2: corresponding homology points in the second camera system
        :param x: initialValues vector by, bz, omega, phi, kappa ( bx=1)

        :type cameraPoints1: np.array nx3
        :type cameraPoints2: np.array nx3
        :type x: np.array (5,1)

        :return: A ,B matrices, w vector

        :rtype: tuple
        """
        numPnts = cameraPoints1.shape[0]  # Number of points

        dbdy = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
        dbdz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])

        dXdx = np.array([1, 0, 0])
        dXdy = np.array([0, 1, 0])

        # Compute rotation matrix and it's derivatives
        rotationMatrix2 = Compute3DRotationMatrix(x[2, 0], x[3, 0], x[4, 0])
        dRdOmega = Compute3DRotationDerivativeMatrix(x[2, 0], x[3, 0], x[4, 0], 'omega')
        dRdPhi = Compute3DRotationDerivativeMatrix(x[2, 0], x[3, 0], x[4, 0], 'phi')
        dRdKappa = Compute3DRotationDerivativeMatrix(x[2, 0], x[3, 0], x[4, 0], 'kappa')

        # Create the skew matrix from the vector [bx, by, bz]
        bMatrix = ComputeSkewMatrixFromVector(np.array([1, x[0, 0], x[1, 0]]))

        # Compute A matrix; the coplanar derivatives with respect to the unknowns by, bz, omega, phi, kappa
        A = np.zeros((numPnts, 5))
        A[:, 0] = np.diag(
            np.dot(cameraPoints1,
                   np.dot(dbdy, np.dot(rotationMatrix2, cameraPoints2.T))))  # derivative in respect to by
        A[:, 1] = np.diag(
            np.dot(cameraPoints1,
                   np.dot(dbdz, np.dot(rotationMatrix2, cameraPoints2.T))))  # derivative in respect to bz
        A[:, 2] = np.diag(
            np.dot(cameraPoints1, np.dot(bMatrix, np.dot(dRdOmega, cameraPoints2.T))))  # derivative in respect to omega
        A[:, 3] = np.diag(
            np.dot(cameraPoints1, np.dot(bMatrix, np.dot(dRdPhi, cameraPoints2.T))))  # derivative in respect to phi
        A[:, 4] = np.diag(
            np.dot(cameraPoints1, np.dot(bMatrix, np.dot(dRdKappa, cameraPoints2.T))))  # derivative in respect to kappa

        # Compute B matrix; the coplanar derivatives in respect to the observations, x', y', x'', y''.
        B = np.zeros((numPnts, 4 * numPnts))
        k = 0
        for i in range(numPnts):
            p1vec = cameraPoints1[i, :]
            p2vec = cameraPoints2[i, :]
            B[i, k] = np.dot(dXdx, np.dot(bMatrix, np.dot(rotationMatrix2, p2vec)))
            B[i, k + 1] = np.dot(dXdy, np.dot(bMatrix, np.dot(rotationMatrix2, p2vec)))
            B[i, k + 2] = np.dot(np.dot(p1vec, np.dot(bMatrix, rotationMatrix2)), dXdx)
            B[i, k + 3] = np.dot(np.dot(p1vec, np.dot(bMatrix, rotationMatrix2)), dXdy)
            k += 4

        # w vector
        w = np.diag(np.dot(cameraPoints1, np.dot(bMatrix, np.dot(rotationMatrix2, cameraPoints2.T))))

        return A, B, w

    def ImagesToModel(self, imagePoints1, imagePoints2, Method):
        """
        Mapping points from image space to model space

        :param imagePoints1: points from the first image
        :param imagePoints2: points from the second image
        :param Method: method for intersection

        :type imagePoints1: np.array nx2
        :type imagePoints2: np.array nx2
        :type Method: string

        :return: corresponding model points
        :rtype: np.array nx3


        .. warning::

            This function is empty, need implementation

        .. note::

            One of the images is a reference, orientation of this image must be set.

        """

    def GroundToImage(self, groundPoints):
        """
        Transforming ground points to image points

        :param groundPoints: ground points [m]

        :type groundPoints: np.array nx3

        :return: corresponding Image points

        :rtype: np.array nx2

        """
        pass

        # 1. calculating pic plane
        # 2. calculating intersection of vector from perspective center to ground poind


    def geometricIntersection(self, cameraPoints1, cameraPoints2):
        """
        Ray Intersection based on geometric calculations.

        :param cameraPoints1: points in the first image
        :param cameraPoints2: corresponding points in the second image

        :type cameraPoints1: np.array nx3
        :type cameraPoints2: np.array nx3

        :return: lambda1, lambda2 scalars

        :rtype: np.array nx2

        .. warning::

            This function is empty, need implementation

        """

    def vectorIntersction(self, cameraPoints1, cameraPoints2):
        """
        Ray Intersection based on vector calculations.

        :param cameraPoints1: points in image space
        :param cameraPoints2: corresponding image points

        :type cameraPoints1: np.array nx
        :type cameraPoints2: np.array nx


        :return: lambda1, lambda2 scalars

        :rtype: np.array nx2

        .. warning::

            This function is empty, need implementation

        """

    def CollinearityIntersection(self, cameraPoints1, cameraPoints2):
        """
        Ray intersection based on the collinearity principle

        :param cameraPoints1: points in image space
        :param cameraPoints2: corresponding image points

        :type cameraPoints1: np.array nx2
        :type cameraPoints2: np.array nx2

        :return: corresponding ground points

        :rtype: np.array nx3

        .. warning::

            This function is empty, need implementation

        """


if __name__ == '__main__':
    camera = Camera(152, None, None, None, None)
    image1 = SingleImage(camera)
    image2 = SingleImage(camera)
    leftCamPnts = np.array([[-4.83, 7.80],
                            [-4.64, 134.86],
                            [5.39, -100.80],
                            [4.58, 55.13],
                            [98.73, 9.59],
                            [62.39, 128.00],
                            [67.90, 143.92],
                            [56.54, -85.76]])
    rightCamPnts = np.array([[-83.17, 6.53],
                             [-102.32, 146.36],
                             [-62.84, -102.87],
                             [-97.33, 56.40],
                             [-3.51, 14.86],
                             [-27.44, 136.08],
                             [-23.70, 152.90],
                             [-8.08, -78.07]])
    new = ImagePair(image1, image2)

    print(new.ComputeDependentRelativeOrientation(leftCamPnts, rightCamPnts, np.array([1, 0, 0, 0, 0, 0])))
