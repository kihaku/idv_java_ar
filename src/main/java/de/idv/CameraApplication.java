package de.idv;

import javafx.application.Application;
import javafx.application.Platform;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.stage.Stage;
import nu.pattern.OpenCV;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.videoio.VideoCapture;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * OpenFx application that grabs the image of a camera and shows inside a window.</br>
 * Transforming the grabbed image is possible with a MatrixProcessor.
 */
public class CameraApplication extends Application {

    private ImageView imageView = null;
    private ScheduledExecutorService scheduler = null;
    private VideoCapture capture = null;
    private MatrixProcessor processor = null;

    public static void main(String[] args) {
        // Initialise OpenCV
        OpenCV.loadShared();
        // Launch the OpenFx application
        launch();
    }

    @Override
    public void start(Stage stage) throws IOException {
        // Create a new MatrixProcessor for analyzing and manipulating the camera image
        this.processor = new MatrixProcessor();

        // Grab the camera
        // Use the correct index if more than one camera exists
        this.capture = new VideoCapture(0);

        // Create the ImageView used for showing the camera image
        this.imageView = new ImageView(grabFrame());
        this.imageView.setFitHeight(400);
        this.imageView.setFitHeight(600);
        this.imageView.setPreserveRatio(true);

        // Add the ImageView to the scene and the scene to the stage (OpenFx)
        Group root = new Group(this.imageView);
        Scene scene = new Scene(root, 600, 400);
        stage.setTitle("IDV AR Camera");
        stage.setScene(scene);
        stage.show();

        // Start a thread to grab and manipulate the camera image
        Runnable frameGrabber = () -> updateImageView(grabFrame());

        // Run the thread every 200 ms (5 frames/sec)
        this.scheduler = Executors.newSingleThreadScheduledExecutor();
        this.scheduler.scheduleAtFixedRate(frameGrabber, 0, 200, TimeUnit.MILLISECONDS);
    }

    @Override
    public void stop() throws Exception {
        super.stop();
        // Stop the grabber thread
        if (this.scheduler != null) {
            this.scheduler.shutdown();
            this.scheduler.awaitTermination(33, TimeUnit.MILLISECONDS);
        }
        // Release the camera
        if (this.capture != null && this.capture.isOpened()) {
            this.capture.release();
        }
    }

    /**
     * Renders the given image on the application's image view
     *
     * @param image the image to show
     */
    private void updateImageView(Image image) {
        Platform.runLater(() -> this.imageView.setImage(image));
    }

    /**
     * Grabs the image from the camera and let the MatrixProcessor process the captured image
     *
     * @return
     */
    private Image grabFrame() {
        if (this.capture.isOpened()) {
            Mat frame = new Mat();
            // Grab the frame from the camera
            if (this.capture.read(frame)) {
                // Let the MatrixProcessor do its work
                frame = this.processor.process(frame);

                // Convert the processed image matrix to a png
                MatOfByte buffer = new MatOfByte();
                Imgcodecs.imencode(".png", frame, buffer);
                return new Image(new ByteArrayInputStream(buffer.toArray()));
            }
        }
        return null;
    }

}
