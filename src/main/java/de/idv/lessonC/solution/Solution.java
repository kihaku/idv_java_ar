package de.idv.lessonC.solution;

import de.idv.Util;
import org.opencv.core.*;
import org.opencv.features2d.BFMatcher;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.ORB;

import java.util.Comparator;
import java.util.List;

public class Solution {

    public Mat process(Mat imageMatrix) {
        // Load the model image
        Mat model = Util.loadFromFile("src/main/resources/model.jpg");

        // Create the ORB detector
        ORB orb = ORB.create();

        // Create brute force matcher
        BFMatcher bfMatcher = BFMatcher.create(BFMatcher.BRUTEFORCE_HAMMING, true);

        // Create a mask
        // <> 0 means every pixel is processed
        Mat mask = new Mat();
        mask.setTo(new Scalar(255));

        // Compute model key points and descriptors
        MatOfKeyPoint modelKeyPoints = new MatOfKeyPoint();
        Mat modelDescriptors = new Mat();
        orb.detectAndCompute(model, mask, modelKeyPoints, modelDescriptors);

        // Compute scene key points and descriptors
        MatOfKeyPoint sceneKeyPoints = new MatOfKeyPoint();
        Mat sceneDescriptors = new Mat();
        orb.detectAndCompute(imageMatrix, mask, sceneKeyPoints, sceneDescriptors);

        // Match scene descriptors with model descriptors
        MatOfDMatch matches = new MatOfDMatch();
        bfMatcher.match(modelDescriptors, sceneDescriptors, matches);

        // Sort by their distance
        List<DMatch> matchList = matches.toList();
        matchList.sort(Comparator.comparing(d -> d.distance));

        // Draw the first/best 15 matches on scene
        Mat processedMat = new Mat();
        MatOfDMatch matOfDMatch = new MatOfDMatch();
        if (matchList.size() > 15) {
            // draw the first 15 matches on scene
            matOfDMatch.fromList(matchList.subList(0, 15));
        }
        // Build matrix from list of DMatch
        matOfDMatch.fromList(matchList);

        // Draw the matches
        // -1 in a scalar means random at this point. so ever color is fully random
        Features2d.drawMatches(model, modelKeyPoints, imageMatrix, sceneKeyPoints, matOfDMatch, processedMat, Scalar.all(-1), Scalar.all(-1), new MatOfByte(), 2);

        return processedMat;
    }

}
