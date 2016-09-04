package edu.byu.rvl.myopencvapp;

import android.app.Activity;
import android.content.Context;
import android.hardware.Camera;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.View;
import android.view.WindowManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import static org.opencv.imgproc.Imgproc.equalizeHist;

public class MyOpenCVActivity extends Activity implements View.OnTouchListener, CvCameraViewListener2 {
    private static final String TAG = "MyOpenCV::Activity";
    private CameraBridgeViewBase mOpenCvCameraView;



    private MenuItem mItemPreviewBili;
    private MenuItem mItemPreviewRGBA;

    public static int pointerCount = 0;

    public static final int      	VIEW_MODE_RGBA      = 0;
    public static final int      	VIEW_MODE_Bili      = 1;
    public static int           	viewMode = VIEW_MODE_RGBA;

    private static final Scalar     FACE_RECT_COLOR     = new Scalar(0, 255, 0, 255);

    boolean mark = false;

    private Mat mRgb;
    private Mat mGray;
    private File mCascadeFile;
    private CascadeClassifier mJavaDetector;


    int FrameHeight;
    int FrameWidth;
    String Msg;
    private int                    mAbsoluteFaceSize   = 0;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");

                    try {
                        // load cascade file from application resources
                        InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
                        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                        mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
                        FileOutputStream os = new FileOutputStream(mCascadeFile);

                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = is.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                        }
                        is.close();
                        os.close();

                        mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                        if (mJavaDetector.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier");
                            mJavaDetector = null;
                        } else
                            Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());

                        cascadeDir.delete();

                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                    }

                    mOpenCvCameraView.enableView();
                    mOpenCvCameraView.setOnTouchListener(MyOpenCVActivity.this);
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        Log.d(TAG, "Creating and setting view");
        mOpenCvCameraView = (CameraBridgeViewBase) new JavaCameraView(this, -1);
        setContentView(mOpenCvCameraView);
        mOpenCvCameraView.setCvCameraViewListener(this);

    }

    @Override
    public void onResume() {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_6, this, mLoaderCallback);
    }

    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        FrameHeight = height;
        FrameWidth = width;
        mGray = new Mat();
        mRgb = new Mat();
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgb.release();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_my_open_cv, menu);
        Log.i(TAG, "called onCreateOptionsMenu");
        mItemPreviewRGBA = menu.add("RGBA");
        mItemPreviewBili = menu.add("Face Detect");
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so lo// as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();
        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
        if (item == mItemPreviewBili) {
            mark = false;
            viewMode= VIEW_MODE_Bili;
        }
        else if(item == mItemPreviewRGBA)
        {
            viewMode= VIEW_MODE_RGBA;
        }

        return super.onOptionsItemSelected(item);
    }

    public boolean onTouch(View v, MotionEvent event) {
        int i;
        pointerCount = event.getPointerCount();
        if (pointerCount == 2)
            mark = true;
        return true;
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        switch(viewMode){
            case VIEW_MODE_Bili:
                if (mark) return mRgb;
                mRgb = inputFrame.rgba();//Get rgb image for display
                mGray = inputFrame.gray();//Get gray image for detection
                MatOfRect faces = new MatOfRect();//Get a rectangle matrix to store the face
                if (mJavaDetector != null){
                    equalizeHist(mGray, mGray);//equalize
                    mJavaDetector.detectMultiScale(mGray, faces, 1.5, 1, 1, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                            new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());}//Detect faces
                Rect[] facesArray = faces.toArray();
                for (int i = 0; i < facesArray.length; i++)
                    Core.rectangle(mRgb, facesArray[i].tl(), facesArray[i].br(), FACE_RECT_COLOR, 3); //Draw rectangles
                Msg = ("Hello Human! numbers:" + facesArray.length);
                Core.putText(mRgb,Msg,new Point(10, 100), 3/* CV_FONT_HERSHEY_COMPLEX */, 2, new Scalar(255, 0, 0, 255), 3);//Print Numbers of persons
                break;
            case VIEW_MODE_RGBA:
                mRgb = inputFrame.rgba();
        }
        return mRgb;
    }
}