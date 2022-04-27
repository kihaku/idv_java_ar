package de.idv.lessonE.solution;

import de.idv.Util;
import de.javagl.obj.*;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.features2d.BFMatcher;
import org.opencv.features2d.ORB;
import org.opencv.imgproc.Imgproc;

import java.io.FileReader;
import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Solution {

    private final Mat model;
    private final MatOfKeyPoint modelKeyPoints;
    private final Mat modelDescriptors;
    private final Mat mask;

    private final Mat cameraMatrix;
    private final Obj obj3d;

    public Solution() throws IOException {
        // Load the model image
        model = Util.loadFromFile("src/main/resources/model.jpg");

        // Create the ORB detector
        ORB orb = ORB.create();

        // Create a mask
        // <> 0 means every pixel is processed
        mask = new Mat();
        mask.setTo(new Scalar(255));

        // Compute model key points and descriptors
        modelKeyPoints = new MatOfKeyPoint();
        modelDescriptors = new Mat();
        orb.detectAndCompute(model, mask, modelKeyPoints, modelDescriptors);

        // Load the cow obj 3d model
        Obj tempObj = ObjReader.read(new FileReader("src/main/resources/cow.obj"));
        obj3d = ObjUtils.convertToRenderable(tempObj);

        // Create the pinhole camera matrix
        // In this case the half of the scene width and height was used (300 and 200)
        // 800 is a constant that should work reasonable with pinhole cameras (distance of the camera)
        cameraMatrix = new Mat(3, 3, CvType.CV_64F);
        cameraMatrix.put(0, 0, 800);
        cameraMatrix.put(0, 1, 0);
        cameraMatrix.put(0, 2, 300);

        cameraMatrix.put(1, 0, 0);
        cameraMatrix.put(1, 1, 800);
        cameraMatrix.put(1, 2, 200);

        cameraMatrix.put(2, 0, 0);
        cameraMatrix.put(2, 1, 0);
        cameraMatrix.put(2, 2, 1);
    }

    public Mat process(Mat imageMatrix) {
        // Create the ORB detector
        ORB orb = ORB.create();

        // Create brute force matcher
        BFMatcher bfMatcher = BFMatcher.create(Core.NORM_HAMMING, true);

        // Compute scene key points and descriptors
        MatOfKeyPoint sceneKeyPoints = new MatOfKeyPoint();
        Mat sceneDescriptors = new Mat();
        orb.detectAndCompute(imageMatrix, mask, sceneKeyPoints, sceneDescriptors);

        // If no descriptions are found, exit the method to prevent hanging/freezing on following methods
        if (sceneDescriptors.rows() == 0 || sceneDescriptors.cols() == 0) {
            return imageMatrix;
        }

        // Match scene descriptors with model descriptors
        MatOfDMatch matches = new MatOfDMatch();
        bfMatcher.match(modelDescriptors, sceneDescriptors, matches);

        // More than 22 matches should be found, otherwise we suppose the model was not found
        List<DMatch> matchList = matches.toList();
        if (matchList.size() > 22) {
            // Build a point matrix of the model key points that were matched
            MatOfPoint2f srcPtsMat = new MatOfPoint2f();
            List<Point> srcPts = matchList.stream().map(m -> modelKeyPoints.toArray()[m.queryIdx].pt).collect(Collectors.toList());
            srcPtsMat.fromList(srcPts);

            // Build a point matrix of the scene key points that were matched
            MatOfPoint2f dstPtsMat = new MatOfPoint2f();
            List<Point> dstPts = matchList.stream().map(m -> sceneKeyPoints.toArray()[m.trainIdx].pt).collect(Collectors.toList());
            dstPtsMat.fromList(dstPts);

            // Compute the homography
            Mat homography = Calib3d.findHomography(srcPtsMat, dstPtsMat, Calib3d.RANSAC, 4);

            // If no homography could be computed, exit the method to prevent hanging/freezing on following methods
            if (homography.rows() == 0 || homography.cols() == 0) {
                return imageMatrix;
            }

            // Create the 3d projection matrix
            Mat projection = createProjectionMatrix(cameraMatrix, homography);
            // Render the 3d model on the scene
            renderObj(imageMatrix, projection);
        }
        return imageMatrix;
    }

    /**
     * Creates the 3d projection matrix with the homography and the pinhole camera matrix
     *
     * @param cameraMatrix pinhole camera matrix
     * @param homography   homography matrix
     * @return the 3d projection matrix
     */
    private Mat createProjectionMatrix(Mat cameraMatrix, Mat homography) {
        // Negate the homography matrix
        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) {
                homography.put(x, y, homography.get(x, y)[0] * -1);
            }
        }

        // Calculate the rotation and translation of the x and y axis
        Mat rotAndTransl = new Mat();
        // Invert the pinhole camera matrix
        Mat invertedCameraMatrix = new Mat(cameraMatrix.rows(), cameraMatrix.cols(), cameraMatrix.type());
        Core.invert(cameraMatrix, invertedCameraMatrix);
        // Multiply the inverted camera matrix with the homography matrix
        Core.gemm(invertedCameraMatrix, homography, 1, new Mat(), 0, rotAndTransl, 0);

        Mat col1 = rotAndTransl.col(0);
        Mat col2 = rotAndTransl.col(1);
        Mat col3 = rotAndTransl.col(2);

        // Normalise the vectors
        double l = Math.sqrt(Core.norm(col1, Core.NORM_L2) * Core.norm(col2, Core.NORM_L2));

        // Divide the first column by the normalised vector to get the first rotation
        Mat rot1 = new Mat(col1.rows(), col1.cols(), col1.type());
        rot1.put(0, 0, col1.get(0, 0)[0] / l);
        rot1.put(1, 0, col1.get(1, 0)[0] / l);
        rot1.put(2, 0, col1.get(2, 0)[0] / l);

        // Divide the Second column by the normalised vector to get the second rotation
        Mat rot2 = new Mat(col2.rows(), col2.cols(), col2.type());
        rot2.put(0, 0, col2.get(0, 0)[0] / l);
        rot2.put(1, 0, col2.get(1, 0)[0] / l);
        rot2.put(2, 0, col2.get(2, 0)[0] / l);

        // Divide the third by the normalised vector to get the translation
        Mat translation = new Mat(col3.rows(), col3.cols(), col3.type());
        translation.put(0, 0, col3.get(0, 0)[0] / l);
        translation.put(1, 0, col3.get(1, 0)[0] / l);
        translation.put(2, 0, col3.get(2, 0)[0] / l);

        // Calculate the orthonormal basis

        // Add rot1 and rot2
        Mat c = new Mat(rot1.rows(), rot1.cols(), rot1.type());
        Core.add(rot1, rot2, c);
        // Calculate the cross product of rot1 and rot2
        Mat p = rot1.cross(rot2);
        // Calculate the cross product of c and p
        Mat d = c.cross(p);

        // Divide the c matrix by the normalised c vector
        Mat normDivC = new Mat(c.rows(), c.cols(), c.type());
        double normC = Core.norm(c, Core.NORM_L2);
        normDivC.put(0, 0, c.get(0, 0)[0] / normC);
        normDivC.put(1, 0, c.get(1, 0)[0] / normC);
        normDivC.put(2, 0, c.get(2, 0)[0] / normC);

        // Divide the d matrix by the normalised d vector
        Mat normDivD = new Mat(d.rows(), d.cols(), d.type());
        double normD = Core.norm(d, Core.NORM_L2);
        normDivD.put(0, 0, d.get(0, 0)[0] / normD);
        normDivD.put(1, 0, d.get(1, 0)[0] / normD);
        normDivD.put(2, 0, d.get(2, 0)[0] / normD);

        // Add the c and d matrix
        Mat addCD = new Mat(normDivC.rows(), normDivC.cols(), normDivC.type());
        Core.add(normDivC, normDivD, addCD);

        // Multiply the sum by 1 / sqrt(2) and save the result in rot1
        Core.multiply(addCD, new Scalar(1 / Math.sqrt(2)), rot1);

        // Subtract the c and d matrix
        Mat subCD = new Mat(normDivC.rows(), normDivC.cols(), normDivC.type());
        Core.subtract(normDivC, normDivD, subCD);

        // Multiply the sum by 1 / sqrt(2) and save the result in rot2
        Core.multiply(subCD, new Scalar(1 / Math.sqrt(2)), rot2);

        // Calculate the cross product of rot1 and rot2
        Mat rot3 = rot1.cross(rot2);

        // Calculate the 3d projection matrix
        // Matrix of rot1, rot2, rot3 and translation
        Mat projection = new Mat(4, 3, CvType.CV_64F);
        projection.put(0, 0, rot1.get(0, 0)[0]);
        projection.put(0, 1, rot1.get(1, 0)[0]);
        projection.put(0, 2, rot1.get(2, 0)[0]);

        projection.put(1, 0, rot2.get(0, 0)[0]);
        projection.put(1, 1, rot2.get(1, 0)[0]);
        projection.put(1, 2, rot2.get(2, 0)[0]);

        projection.put(2, 0, rot3.get(0, 0)[0]);
        projection.put(2, 1, rot3.get(1, 0)[0]);
        projection.put(2, 2, rot3.get(2, 0)[0]);

        projection.put(3, 0, translation.get(0, 0)[0]);
        projection.put(3, 1, translation.get(1, 0)[0]);
        projection.put(3, 2, translation.get(2, 0)[0]);

        // Transpose the matrix
        projection = projection.t();

        // Multiply the pinhole camera matrix with the transposed projection matrix
        Mat result = new Mat();
        Core.gemm(cameraMatrix, projection, 1, new Mat(), 0, result, 0);

        return result;
    }

    private void renderObj(Mat scene, Mat projection) {
        final int h = model.height();
        final int w = model.width();

        double modelScaling = 0.25;

        // Draw every face of the 3d obj model
        IntStream.range(0, obj3d.getNumFaces()).boxed().forEach(i -> {
            // Get the face
            final ObjFace face = obj3d.getFace(i);
            // Get all vertices of the face
            final List<Integer> faceVertices = IntStream.range(0, face.getNumVertices()).boxed().map(face::getVertexIndex).collect(Collectors.toList());

            // Build a matrix of the vertices points
            final Mat vertexPoints = new Mat(faceVertices.size(), 1, CvType.CV_64FC3);
            int row = 0;
            for (Integer vi : faceVertices) {
                FloatTuple vertex = obj3d.getVertex(vi);
                // Add the half of weight and height to center the 3d model
                vertexPoints.put(row++, 0,
                        (vertex.getX() * modelScaling) + (w / 2.0),
                        (vertex.getZ() * modelScaling) + (h / 2.0),
                        vertex.getY() * modelScaling);
            }

            // Transform the plan marked by the face vertices to match the perspective of the scene
            Mat transformed = new Mat();
            Core.perspectiveTransform(vertexPoints, transformed, projection);

            // Create the points of the transformed matrix
            Point[] pointArr = IntStream.range(0, faceVertices.size()).boxed().map(pi -> new Point(transformed.get(pi, 0)[0], transformed.get(pi, 0)[1])).toArray(Point[]::new);

            // We don't want to render the model if a point is negative as it would overdraw the whole screen
            for (Point point : pointArr) {
                if (point.x < 0 || point.y < 0) {
                    // No rendering when a negative value exists
                    return;
                }
            }

            // Draw the vertices
            Imgproc.fillConvexPoly(scene, new MatOfPoint(pointArr), new Scalar(182, 170, 58));
        });
    }
}
