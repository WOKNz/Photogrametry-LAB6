import numpy as np
from Camera import Camera
from MatrixMethods import Compute3DRotationMatrix, Compute3DRotationDerivativeMatrix
import numpy.linalg as la

class SingleImage(object):

    def __init__(self, camera):
        """
        Initialize the SingleImage object

        :param camera: instance of the Camera class
        :param points: points in image space

        :type camera: Camera
        :type points: np.array

        """
        self.__camera = camera
        self.__innerOrientationParameters = None
        self.__isSolved = False
        self.__exteriorOrientationParameters = np.array([0, 0, 0, 0, 0, 0], 'f')
        self.__rotationMatrix = None

    @property
    def innerOrientationParameters(self):
        """
        Inner orientation parameters


        .. warning::

            Can be held either as dictionary or array. For your implementation and decision.

        .. note::

            Do not forget to decide how it is held and document your decision

        :return: inner orinetation parameters

        :rtype: **ADD**
        """
        return self.__innerOrientationParameters

    @property
    def camera(self):
        """
        The camera that took the image

        :rtype: Camera

        """
        return self.__camera

    @property
    def exteriorOrientationParameters(self):
        r"""
        Property for the exterior orientation parameters

        :return: exterior orientation parameters in the following order, **however you can decide how to hold them (dictionary or array)**

        .. math::
            exteriorOrientationParameters = \begin{bmatrix} X_0 \\ Y_0 \\ Z_0 \\ \omega \\ \varphi \\ \kappa \end{bmatrix}

        :rtype: np.ndarray or dict
        """
        return self.__exteriorOrientationParameters

    @exteriorOrientationParameters.setter
    def exteriorOrientationParameters(self, parametersArray):
        r"""

        :param parametersArray: the parameters to update the ``self.__exteriorOrientationParameters``

        **Usage example**

        .. code-block:: py

            self.exteriorOrintationParameters = parametersArray

        """
        self.__exteriorOrientationParameters = parametersArray

    @property
    def rotationMatrix(self):
        """
        The rotation matrix of the image

        Relates to the exterior orientation
        :return: rotation matrix

        :rtype: np.ndarray (3x3)
        """

        R = Compute3DRotationMatrix(self.exteriorOrientationParameters[3], self.exteriorOrientationParameters[4],
                                    self.exteriorOrientationParameters[5])

        return R

    @property
    def isSolved(self):
        """
        True if the exterior orientation is solved

        :return True or False

        :rtype: boolean
        """
        return self.__isSolved

    def ComputeInnerOrientation(self, imagePoints):
        """
        Compute inner orientation parameters

        :param imagePoints: coordinates in image space

        :type imagePoints: np.array nx2

        :return: Inner orientation parameters, their accuracies, and the residuals vector

        :rtype: dict


        .. note::

            - Don't forget to update the ``self.__innerOrinetationParameters`` member. You decide the type
            - The fiducial marks are held within the camera attribute of the object, i.e., ``self.camera.fiducialMarks``
            - return values can be a tuple of dictionaries and arrays.

        **Usage example**

        .. code-block:: py

            fMarks = np.array([[113.010, 113.011],
                       [-112.984, -113.004],
                       [-112.984, 113.004],
                       [113.024, -112.999]])
            img_fmarks = np.array([[-7208.01, 7379.35],
                        [7290.91, -7289.28],
                        [-7291.19, -7208.22],
                        [7375.09, 7293.59]])
            cam = Camera(153.42, np.array([0.015, -0.020]), None, None, fMarks)
            img = SingleImage(camera = cam, points = None)
            inner_parameters, accuracies, residuals = img.ComputeInnerOrientation(img_fmarks)
        """

        def a(l0, varnum):  # Calculating Matrix A for Least Square
            result = np.zeros((imagePoints.shape[0] * 2, varnum))
            result[range(0, result.shape[0], 2), 0] = 1
            result[range(1, result.shape[0], 2), 1] = 1
            result[range(0, result.shape[0], 2), 2:4] = l0[:, :]
            result[range(1, result.shape[0], 2), 4:6] = l0[:, :]
            return result

        def La(l1):
            result = np.zeros((l1.shape[0] * 2, 1))
            # print(result)
            for i in range(l1.shape[0]):
                result[i * 2, 0] = l1[i, 0]
                result[i * 2 + 1, 0] = l1[i, 1]
            return result

        def n(lstqA):
            return np.dot(lstqA.T, lstqA)

        def u(lstqA, lstqL):
            return np.dot(lstqA.T, lstqL)

        def ansx(lstqN, lstqu):
            return np.dot(np.linalg.inv(lstqN), lstqu)

        lstqA = a(self.camera.fiducialMarks[:imagePoints.shape[0],:], 6)
        #print('A:', lstqA)
        lstqN = n(lstqA)
        #print('N:', lstqN)
        lstqL = La(imagePoints)
        # print('L:', lstqL)
        lstqu = u(lstqA, lstqL)
        # print('U:', lstqu)
        ans = ansx(lstqN, lstqu)
        # print('Ans:', ans)
        v = np.dot(lstqA, ans) - lstqL
        n_inv = np.linalg.inv(lstqN)

        self.__innerOrientationParameters = ans
        return ans, v, n_inv;

    def ComputeGeometricParameters(self, params):
        """
        Computes the geometric inner orientation parameters

        :return: geometric inner orientation parameters

        :rtype: dict

        .. warning::

           This function is empty, need implementation

        .. note::

            The algebraic inner orinetation paramters are held in ``self.innerOrientatioParameters`` and their type
            is according to what you decided when initialized them

        """
        a0, b0, a1, a2, b1, b2 = params[:]
        translationX = a0
        translationY = b0
        rotationAngle = np.arctan(b1 / b2)
        shearAngle = np.arctan((a1 * np.sin(rotationAngle) + a2 * np.cos(rotationAngle)) / \
                               (b1 * np.sin(rotationAngle) + b2 * np.cos(rotationAngle)))
        scaleFactorX = a1 * np.cos(rotationAngle) - a2 * np.sin(rotationAngle)
        scaleFactorY = a1 * np.sin(rotationAngle) + a2 * np.cos(rotationAngle) / \
                       np.sin(shearAngle)

        return {'Dx': translationX,
                'Dy': translationY,
                'Thetha': rotationAngle,
                'Gamma': shearAngle,
                'Sx': scaleFactorX,
                'Sy': scaleFactorY}

    def ComputeInverseInnerOrientation(self, imagePoints):
        """
        Computes the parameters of the inverse inner orientation transformation

        :return: parameters of the inverse transformation

        :rtype: dict

        .. warning::

            This function is empty, need implementation

        .. note::

            The inner orientation algebraic parameters are held in ``self.innerOrientationParameters``
            their type is as you decided when implementing
        """

        # def a(l0, varnum):  # Calculating Matrix A for Least Square
        #     result = np.zeros((l0.shape[0] * 2, varnum))
        #     result[range(0, result.shape[0], 2), 0] = 1
        #     result[range(1, result.shape[0], 2), 1] = 1
        #     result[range(0, result.shape[0], 2), 2:4] = l0[:, :]
        #     result[range(1, result.shape[0], 2), 4:6] = l0[:, :]
        #     return result
        #
        # def La(l1):
        #     result = np.zeros((l1.shape[0] * 2, 1))
        #     # print(result)
        #     for i in range(l1.shape[0]):
        #         result[i * 2, 0] = l1[i, 0]
        #         result[i * 2 + 1, 0] = l1[i, 1]
        #     return result
        #
        # def n(lstqA):
        #     return np.dot(lstqA.T, lstqA)
        #
        # def u(lstqA, lstqL):
        #     return np.dot(lstqA.T, lstqL)
        #
        # def ansx(lstqN, lstqu):
        #     return np.dot(np.linalg.inv(lstqN), lstqu)
        #
        # lstqA = a(imagePoints, 6)
        # # print('A:', lstqA)
        # lstqN = n(lstqA)
        # # print('N:', lstqN)
        # lstqL = La(self.camera.fiducialMarks)
        # # print('L:', lstqL)
        # lstqu = u(lstqA, lstqL)
        # # print('U:', lstqu)
        # ans = ansx(lstqN, lstqu)
        # # print('Ans:', ans)
        # v = np.dot(lstqA, ans) - lstqL
        # n_inv = np.linalg.inv(lstqN)
        # return ans, v, n_inv;

        a0,b0,a1,a2,b1,b2 = self.innerOrientationParameters[:]

        afin_matrix = la.inv(np.array([[a1,a2], [b1, b2]]))

        return np.array([a0, b0, afin_matrix[0,0],\
                         afin_matrix[0,1],\
                         afin_matrix[1,0],\
                         afin_matrix[1,1]])

    def CameraToImage(self, cameraPoints):
        """
        Transforms camera points to image points

        :param cameraPoints: camera points

        :type cameraPoints: np.array nx2

        :return: corresponding Image points

        :rtype: np.array nx2


        .. warning::

            This function is empty, need implementation

        .. note::

            The inner orientation parameters required for this function are held in ``self.innerOrientationParameters``

        **Usage example**

        .. code-block:: py

            fMarks = np.array([[113.010, 113.011],
                       [-112.984, -113.004],
                       [-112.984, 113.004],
                       [113.024, -112.999]])
            img_fmarks = np.array([[-7208.01, 7379.35],
                        [7290.91, -7289.28],
                        [-7291.19, -7208.22],
                        [7375.09, 7293.59]])
            cam = Camera(153.42, np.array([0.015, -0.020]), None, None, fMarks)
            img = SingleImage(camera = cam, points = None)
            img.ComputeInnerOrientation(img_fmarks)
            pts_image = img.Camera2Image(fMarks)

        """
        a0, b0, a1, a2, b1, b2 = self.__innerOrientationParameters[:]
        dx = np.array([[a0], [b0]])
        r = np.array([[a1, a2], [b1, b2]])
        result = np.zeros(cameraPoints.shape)
        for i in range(0, cameraPoints.shape[0], 1):
            result[i, :] = (dx + np.dot(r, cameraPoints[i, :].reshape(2, 1))).T
        return result

    def ImageToCamera(self, imagePoints):
        """

        Transforms image points to ideal camera points

        :param imagePoints: image points

        :type imagePoints: np.array nx2

        :return: corresponding camera points

        :rtype: np.array nx2

        .. warning::

            This function is empty, need implementation

        .. note::

            The inner orientation parameters required for this function are held in ``self.innerOrientationParameters``


        **Usage example**

        .. code-block:: py

            fMarks = np.array([[113.010, 113.011],
                       [-112.984, -113.004],
                       [-112.984, 113.004],
                       [113.024, -112.999]])
            img_fmarks = np.array([[-7208.01, 7379.35],
                        [7290.91, -7289.28],
                        [-7291.19, -7208.22],
                        [7375.09, 7293.59]])
            cam = Camera(153.42, np.array([0.015, -0.020]), None, None, fMarks)
            img = SingleImage(camera = cam, points = None)
            img.ComputeInnerOrientation(img_fmarks)
            pts_camera = img.Image2Camera(img_fmarks)

        """
        a0, b0, a1, a2, b1, b2 = self.__innerOrientationParameters[:]
        dx = np.array([[a0[0]], [b0[0]]])
        r = np.array([[a1[0], a2[0]], [b1[0], b2[0]]])
        result = np.zeros(imagePoints.shape)
        for i in range(0, imagePoints.shape[0], 1):
            result[i, :] = (np.dot(np.linalg.inv(r), imagePoints[i, :].reshape(2, 1)-dx)).T
        return result

    def ComputeExteriorOrientation(self, imagePoints, groundPoints, epsilon):
        """
        Compute exterior orientation parameters.

        This function can be used in conjecture with ``self.__ComputeDesignMatrix(groundPoints)`` and ``self__ComputeObservationVector(imagePoints)``

        :param imagePoints: image points
        :param groundPoints: corresponding ground points

            .. note::

                Angles are given in radians

        :param epsilon: threshold for convergence criteria

        :type imagePoints: np.array nx2
        :type groundPoints: np.array nx3
        :type epsilon: float

        :return: Exterior orientation parameters: tuple (ndarray (X0, Y0, Z0, omega, phi, kappa), their accuracies, and residuals vector).
        :rtype: tuple


        .. warning::

           - This function is empty, need implementation
           - Decide how the parameters are held, don't forget to update documentation

        .. note::

            - Don't forget to update the ``self.exteriorOrientationParameters`` member (every iteration and at the end).
            - Don't forget to call ``cameraPoints = self.ImageToCamera(imagePoints)`` to correct the coordinates              that are sent to ``self.__ComputeApproximateVals(cameraPoints, groundPoints)``
            - return values can be a tuple of dictionaries and arrays.

        **Usage Example**

        .. code-block:: py

            img = SingleImage(camera = cam)
            grdPnts = np.array([[201058.062, 743515.351, 243.987],
                        [201113.400, 743566.374, 252.489],
                        [201112.276, 743599.838, 247.401],
                        [201166.862, 743608.707, 248.259],
                        [201196.752, 743575.451, 247.377]])
            imgPnts3 = np.array([[-98.574, 10.892],
                         [-99.563, -5.458],
                         [-93.286, -10.081],
                         [-99.904, -20.212],
                         [-109.488, -20.183]])
            img.ComputeExteriorOrientation(imgPnts3, grdPnts, 0.3)

        """
        x_abs = epsilon+1
        cameraPoints = self.ImageToCamera(imagePoints)
        self.__ComputeApproximateVals(cameraPoints, groundPoints)
        while x_abs > epsilon:
            a = self.__ComputeDesignMatrix(groundPoints)
            l0 = self.__ComputeObservationVector(groundPoints)
            l = (cameraPoints.reshape(cameraPoints.size,1).T - l0).reshape(cameraPoints.size,1)
            n = np.dot(a.T,a)
            u = np.dot(a.T,l)
            dx = np.squeeze(np.dot(np.linalg.inv(n),u), axis=1)
            x_abs = np.linalg.norm(dx)
            self.exteriorOrientationParameters += dx
        v = l - cameraPoints.reshape(cameraPoints.size,1)/1000
        if a.shape[0]-6 > 0:
            rms = np.dot(v.T,v)/(a.shape[0]-6)
        else:
            rms = np.dot(v.T, v)
        return self.exteriorOrientationParameters,(np.diag(rms*np.linalg.inv(n)))**0.5/1000,v

    def GroundToImage(self, groundPoints):
        """
        Transforming ground points to image points

        :param groundPoints: ground points [m]

        :type groundPoints: np.array nx3

        :return: corresponding Image points

        :rtype: np.array nx2

        """
        pass  # delete after implementation

    def ImageToRay(self, imagePoints):
        """
        Transforms Image point to a Ray in world system

        :param imagePoints: coordinates of an image point

        :type imagePoints: np.array nx2

        :return: Ray direction in world system

        :rtype: np.array nx3

        .. warning::

           This function is empty, need implementation

        .. note::

            The exterior orientation parameters needed here are called by ``self.exteriorOrientationParameters``
        """
        pass  # delete after implementations

    def ImageToGround_GivenZ(self, imagePoints, Z_values):
        """
        Compute corresponding ground point given the height in world system

        :param imagePoints: points in image space
        :param Z_values: height of the ground points


        :type Z_values: np.array nx1
        :type imagePoints: np.array nx2
        :type eop: np.ndarray 6x1

        :return: corresponding ground points

        :rtype: np.ndarray

        .. warning::

             This function is empty, need implementation

        .. note::

            - The exterior orientation parameters needed here are called by ``self.exteriorOrientationParameters``
            - The focal length can be called by ``self.camera.focalLength``

        **Usage Example**

        .. code-block:: py


            imgPnt = np.array([-50., -33.])
            img.ImageToGround_GivenZ(imgPnt, 115.)

        """
        # print('ImagePoints:',imagePoints,'\nZ_Values:',Z_values)
        imgpoint = np.array([[imagePoints[0],\
                              imagePoints[1],\
                              -self.camera.focalLength]]).reshape(3,1)
        # print('\nImgpoints:',imgpoint)
        r = Compute3DRotationMatrix(self.exteriorOrientationParameters[0],\
                                    self.exteriorOrientationParameters[1],\
                                    self.exteriorOrientationParameters[2])
        lam = (Z_values-self.exteriorOrientationParameters[2])/(np.dot(r,imgpoint))[2]
        # print('\nLam:',lam)
        return (self.exteriorOrientationParameters[0:3]).reshape(3,1)+lam*np.dot(r,imgpoint)


    # ---------------------- Private methods ----------------------

    def __ComputeApproximateVals(self, cameraPoints, groundPoints):
        """
        Compute exterior orientation approximate values via 2-D conform transformation

        :param cameraPoints: points in image space (x y)
        :param groundPoints: corresponding points in world system (X, Y, Z)

        :type cameraPoints: np.ndarray [nx2]
        :type groundPoints: np.ndarray [nx3]

        :return: Approximate values of exterior orientation parameters
        :rtype: np.ndarray

        .. note::

            - ImagePoints should be transformed to ideal camera using ``self.ImageToCamera(imagePoints)``. See code below
            - The focal length is stored in ``self.camera.focalLength``
            - Don't forget to update ``self.exteriorOrientationParameters`` in the order defined within the property
            - return values can be a tuple of dictionaries and arrays.
        """

        # Find approximate values

        A = np.array([[1,0,cameraPoints[0,0],cameraPoints[0,1]],\
                      [0,1,cameraPoints[0,1],-cameraPoints[0,0]],\
                      [1,0,cameraPoints[1,0],cameraPoints[1,1]],\
                      [0,1,cameraPoints[1,1],-cameraPoints[1,0]]])

        b = np.array([[groundPoints[0,0],groundPoints[0,1],groundPoints[1,0],groundPoints[1,1]]]).T
        x = np.dot(np.linalg.inv(A),b)
        lam = np.sqrt(x[2]**2+x[3]**2)
        k = np.arctan2(-(x[3]),(x[2]))
        x0,y0,z0 = x[0],x[1],groundPoints[0,2]+lam*self.camera.focalLength
        omega,phi = 0,0

        self.exteriorOrientationParameters = np.array([x0,y0,z0,omega,phi,k])

        return self.exteriorOrientationParameters

    def __ComputeObservationVector(self, groundPoints):
        """
        Compute observation vector for solving the exterior orientation parameters of a single image
        based on their approximate values

        :param groundPoints: Ground coordinates of the control points

        :type groundPoints: np.array nx3

        :return: Vector l0

        :rtype: np.array nx1
        """

        n = groundPoints.shape[0]  # number of points

        # Coordinates subtraction
        dX = groundPoints[:, 0] - self.exteriorOrientationParameters[0]
        dY = groundPoints[:, 1] - self.exteriorOrientationParameters[1]
        dZ = groundPoints[:, 2] - self.exteriorOrientationParameters[2]
        dXYZ = np.vstack([dX, dY, dZ])
        rotated_XYZ = np.dot(self.rotationMatrix.T, dXYZ).T

        l0 = np.empty(n * 2)

        # Computation of the observation vector based on approximate exterior orientation parameters:
        l0[::2] = -self.camera.focalLength * rotated_XYZ[:, 0] / rotated_XYZ[:, 2]
        l0[1::2] = -self.camera.focalLength * rotated_XYZ[:, 1] / rotated_XYZ[:, 2]

        return l0

    def __ComputeDesignMatrix(self, groundPoints):
        """
            Compute the derivatives of the collinear law (design matrix)

            :param groundPoints: Ground coordinates of the control points

            :type groundPoints: np.array nx3

            :return: The design matrix

            :rtype: np.array nx6

        """
        # initialization for readability
        omega = self.exteriorOrientationParameters[3]
        phi = self.exteriorOrientationParameters[4]
        kappa = self.exteriorOrientationParameters[5]

        # Coordinates subtraction
        dX = groundPoints[:, 0] - self.exteriorOrientationParameters[0]
        dY = groundPoints[:, 1] - self.exteriorOrientationParameters[1]
        dZ = groundPoints[:, 2] - self.exteriorOrientationParameters[2]
        dXYZ = np.vstack([dX, dY, dZ])

        rotationMatrixT = self.rotationMatrix.T
        rotatedG = rotationMatrixT.dot(dXYZ)
        rT1g = rotatedG[0, :]
        rT2g = rotatedG[1, :]
        rT3g = rotatedG[2, :]

        focalBySqauredRT3g = self.camera.focalLength / rT3g ** 2

        dxdg = rotationMatrixT[0, :][None, :] * rT3g[:, None] - rT1g[:, None] * rotationMatrixT[2, :][None, :]
        dydg = rotationMatrixT[1, :][None, :] * rT3g[:, None] - rT2g[:, None] * rotationMatrixT[2, :][None, :]

        dgdX0 = np.array([-1, 0, 0], 'f')
        dgdY0 = np.array([0, -1, 0], 'f')
        dgdZ0 = np.array([0, 0, -1], 'f')

        # Derivatives with respect to X0
        dxdX0 = -focalBySqauredRT3g * np.dot(dxdg, dgdX0)
        dydX0 = -focalBySqauredRT3g * np.dot(dydg, dgdX0)

        # Derivatives with respect to Y0
        dxdY0 = -focalBySqauredRT3g * np.dot(dxdg, dgdY0)
        dydY0 = -focalBySqauredRT3g * np.dot(dydg, dgdY0)

        # Derivatives with respect to Z0
        dxdZ0 = -focalBySqauredRT3g * np.dot(dxdg, dgdZ0)
        dydZ0 = -focalBySqauredRT3g * np.dot(dydg, dgdZ0)

        dRTdOmega = Compute3DRotationDerivativeMatrix(omega, phi, kappa, 'omega').T
        dRTdPhi = Compute3DRotationDerivativeMatrix(omega, phi, kappa, 'phi').T
        dRTdKappa = Compute3DRotationDerivativeMatrix(omega, phi, kappa, 'kappa').T

        gRT3g = dXYZ * rT3g

        # Derivatives with respect to Omega
        dxdOmega = -focalBySqauredRT3g * (dRTdOmega[0, :][None, :].dot(gRT3g) -
                                          rT1g * (dRTdOmega[2, :][None, :].dot(dXYZ)))[0]

        dydOmega = -focalBySqauredRT3g * (dRTdOmega[1, :][None, :].dot(gRT3g) -
                                          rT2g * (dRTdOmega[2, :][None, :].dot(dXYZ)))[0]

        # Derivatives with respect to Phi
        dxdPhi = -focalBySqauredRT3g * (dRTdPhi[0, :][None, :].dot(gRT3g) -
                                        rT1g * (dRTdPhi[2, :][None, :].dot(dXYZ)))[0]

        dydPhi = -focalBySqauredRT3g * (dRTdPhi[1, :][None, :].dot(gRT3g) -
                                        rT2g * (dRTdPhi[2, :][None, :].dot(dXYZ)))[0]

        # Derivatives with respect to Kappa
        dxdKappa = -focalBySqauredRT3g * (dRTdKappa[0, :][None, :].dot(gRT3g) -
                                          rT1g * (dRTdKappa[2, :][None, :].dot(dXYZ)))[0]

        dydKappa = -focalBySqauredRT3g * (dRTdKappa[1, :][None, :].dot(gRT3g) -
                                          rT2g * (dRTdKappa[2, :][None, :].dot(dXYZ)))[0]

        # all derivatives of x and y
        dd = np.array([np.vstack([dxdX0, dxdY0, dxdZ0, dxdOmega, dxdPhi, dxdKappa]).T,
                       np.vstack([dydX0, dydY0, dydZ0, dydOmega, dydPhi, dydKappa]).T])

        a = np.zeros((2 * dd[0].shape[0], 6))
        a[0::2] = dd[0]
        a[1::2] = dd[1]

        return a


if __name__ == '__main__':
    fMarks = np.array([[113.010, 113.011],
                       [-112.984, -113.004],
                       [-112.984, 113.004],
                       [113.024, -112.999]])
    img_fmarks = np.array([[-7208.01, 7379.35],
                           [7290.91, -7289.28],
                           [-7291.19, -7208.22],
                           [7375.09, 7293.59]])
    cam = Camera(153.42, np.array([0.015, -0.020]), None, None, fMarks)
    img = SingleImage(camera=cam)
    print(img.ComputeInnerOrientation(img_fmarks))

    print(img.ImageToCamera(img_fmarks))

    print(img.CameraToImage(fMarks))

    GrdPnts = np.array([[5100.00, 9800.00, 100.00]])
    print(img.GroundToImage(GrdPnts))

    imgPnt = np.array([23.00, 25.00])
    print(img.ImageToRay(imgPnt))

    imgPnt2 = np.array([-50., -33.])
    print(img.ImageToGround_GivenZ(imgPnt2, 115.))

    # grdPnts = np.array([[201058.062, 743515.351, 243.987],
    #                     [201113.400, 743566.374, 252.489],
    #                     [201112.276, 743599.838, 247.401],
    #                     [201166.862, 743608.707, 248.259],
    #                     [201196.752, 743575.451, 247.377]])
    #
    # imgPnts3 = np.array([[-98.574, 10.892],
    #                      [-99.563, -5.458],
    #                      [-93.286, -10.081],
    #                      [-99.904, -20.212],
    #                      [-109.488, -20.183]])
    #
    # intVal = np.array([200786.686, 743884.889, 954.787, 0, 0, 133 * np.pi / 180])
    #
    # print img.ComputeExteriorOrientation(imgPnts3, grdPnts, intVal)
