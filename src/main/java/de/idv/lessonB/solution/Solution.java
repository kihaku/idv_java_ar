package de.idv.lessonB.solution;

import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Scalar;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.ORB;

public class Solution {

    public Mat process(Mat imageMatrix) {
        // Create the ORB detector
        ORB orb = ORB.create();

        // Find key points with the ORB
        MatOfKeyPoint matOfKeyPoint = new MatOfKeyPoint();
        orb.detect(imageMatrix, matOfKeyPoint);

        // Compute the descriptors with ORB
        Mat descriptors = new Mat();
        orb.compute(imageMatrix, matOfKeyPoint, descriptors);

        // Draw the key points on the image
        Mat processedMat = new Mat();
        Features2d.drawKeypoints(imageMatrix, matOfKeyPoint, processedMat, new Scalar(0, 255, 0));

        return processedMat;
    }

}
