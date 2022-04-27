package de.idv;

import javafx.scene.image.Image;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.ByteArrayInputStream;

public class Util {

    private Util() {}

    public static Mat loadFromFile(String imagePath) {
        return Imgcodecs.imread(imagePath);
    }

    public static void saveToFile(Mat imageMatrix, String targetPath) {
        Imgcodecs.imwrite(targetPath, imageMatrix);
    }

    public static Image convertToImage(Mat frame) {
        MatOfByte buffer = new MatOfByte();
        Imgcodecs.imencode(".png", frame, buffer);
        return new Image(new ByteArrayInputStream(buffer.toArray()));
    }
}
