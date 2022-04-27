package de.idv.lessonA;

import de.idv.Util;
import nu.pattern.OpenCV;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Scalar;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.ORB;

/**
 * Class to show how to find key points and mark them on a given image
 */
public class KeyPoints {

    public static void main(String[] args) {
        if (args.length != 2) {
            System.out.println("Usage: IdvAR <input-file> <output-file>");
            System.exit(1);
        }

        // Initialise OpenCV
        OpenCV.loadShared();

        // Load the source image
        Mat loadedImage = Util.loadFromFile(args[0]);

        // Create the ORB detector
        ORB orb = ORB.create();

        // Find key points with the ORB
        MatOfKeyPoint matOfKeyPoint = new MatOfKeyPoint();
        orb.detect(loadedImage, matOfKeyPoint);

        // Draw the key points on the image
        Mat output = new Mat();
        Features2d.drawKeypoints(loadedImage, matOfKeyPoint, output, new Scalar(0, 255, 0));

        // Save the result to the disk
        Util.saveToFile(output, args[1]);
    }
}
