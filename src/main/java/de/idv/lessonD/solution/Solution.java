package de.idv.lessonD.solution;

import de.idv.Util;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.features2d.BFMatcher;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.ORB;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Solution {

    private final Mat model;
    private final MatOfKeyPoint modelKeyPoints;
    private final Mat modelDescriptors;
    private final Mat mask;

    // Do this only one time to get smoothed matches
    public Solution() {
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

        // Draw all matches
        Mat processedMat = new Mat();
        MatOfByte drawnMatches = new MatOfByte();
        // -1 in a scalar means random at this point. so ever color is fully random
        Features2d.drawMatches(model, modelKeyPoints, imageMatrix, sceneKeyPoints, matches, processedMat, Scalar.all(-1), Scalar.all(-1), drawnMatches, 2);

        // More than 15 matches should be found, otherwise we suppose the model was not found
        List<DMatch> matchList = matches.toList();

        if (matchList.size() > 15) {
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
                return processedMat;
            }

            // Draw a rectangle that marks the found model
            int height = model.height();
            int width = model.width();

            // The corner points of the model image
            final Mat modelCorners = new Mat(4, 1, CvType.CV_32FC2);
            modelCorners.put(0, 0, 0, 0);
            modelCorners.put(1, 0, width - 1, 0);
            modelCorners.put(2, 0, width - 1, height - 1);
            modelCorners.put(3, 0, 0, height - 1);

            // Transform the plan marked by the corner points to match the perspective of the scene
            Mat transformed = new Mat(4, 1, CvType.CV_32FC2);
            Core.perspectiveTransform(modelCorners, transformed, homography);

            // Create a matrix of points for drawing
            List<MatOfPoint> list = new ArrayList<>();
            list.add(new MatOfPoint(
                            IntStream.range(0, 4).boxed().map(i -> new Point(transformed.get(i, 0)[0] + width, transformed.get(i, 0)[1])).toArray(Point[]::new)
                    )
            );

            // Draw the rectangle on screen to mark the found model
            Imgproc.polylines(processedMat, list, true, new Scalar(255, 0, 0), 3, Imgproc.LINE_AA);
        }
        return processedMat;
    }

}
