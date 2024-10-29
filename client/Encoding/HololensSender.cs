using System;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using UnityEngine.Windows.WebCam;
using TMPro;

//? This script enables Hololens 2 to send captured images to the server
public class HololensSender : MonoBehaviour
{
    private PhotoCapture photoCaptureObject = null;
    private CameraParameters cameraParameters;
    private Texture2D targetTexture;
    private TcpClient client;
    private NetworkStream stream;
    private object lockObject = new object();

    private void Start()
    {
        ConnectToServer("0.0.0.0", 9999); //! Replace with your server's IP address and port
        //my_text.text += "\n Connected!";

        PhotoCapture.CreateAsync(false, delegate (PhotoCapture captureObject)
        {
            photoCaptureObject = captureObject;
            cameraParameters = new CameraParameters
            {
                hologramOpacity = 0.0f,
                cameraResolutionWidth = 1280,
                cameraResolutionHeight = 720,
                pixelFormat = CapturePixelFormat.BGRA32
            };

            targetTexture = new Texture2D(1280, 720, TextureFormat.BGRA32, false);

            photoCaptureObject.StartPhotoModeAsync(cameraParameters, delegate (PhotoCapture.PhotoCaptureResult result)
            {
                if (result.success)
                {
                    //my_text.text += "\n Photo mode started successfully";
                    Debug.Log("Photo mode started successfully");
                    InvokeRepeating("CaptureFrame", 0f, 0.5f); // Capture 4 frames per second
                }
                else
                {
                    Debug.LogError("Failed to start photo mode: " + result.hResult);
                    //my_text.text += "\n Failed to start photo mode: " + result.hResult;
                }
            });
        });
    }

    private void ConnectToServer(string serverIP, int port)
    {
        try
        {
            client = new TcpClient(serverIP, port);
            stream = client.GetStream();
            Debug.Log("Connected to server");
        }
        catch (Exception ex)
        {
            Debug.LogError("Error connecting to server: " + ex.Message);
        }
    }

    private void CaptureFrame()
    {
        try
        {
            photoCaptureObject.TakePhotoAsync(OnCapturedPhotoToMemory);
            //my_text.text += "\n Photo capture here";
        }
        catch (Exception ex)
        {
            Debug.LogError("Error in CaptureFrame: " + ex.Message);
            //my_text.text += "\n Error in CaptureFrame: " + ex.Message;
        }
    }

    void OnCapturedPhotoToMemory(PhotoCapture.PhotoCaptureResult result, PhotoCaptureFrame photoCaptureFrame)
    {
        if (result.success)
        {
            Debug.Log("Photo captured successfully");

            if (targetTexture == null)
            {
                Debug.LogError("Target texture is null. Initializing it now.");
                //my_text.text += "\n Target texture is null. Initializing it now.";
                targetTexture = new Texture2D(1280, 720, TextureFormat.BGRA32, false);
            }

            photoCaptureFrame.UploadImageDataToTexture(targetTexture);
            byte[] imageBytes = targetTexture.EncodeToPNG(); // Keep using PNG encoding
            Debug.Log("Encoded image data: " + imageBytes.Length + " bytes");

            if (imageBytes != null && imageBytes.Length > 0)
            {
                Thread sendThread = new Thread(() => SendFrame(imageBytes));
                sendThread.Start();
            }
            else
            {
                Debug.LogError("Image buffer is empty or null");
            }
        }
        else
        {
            Debug.LogError("Failed to capture photo: " + result.hResult);
        }
    }

    private void SendFrame(byte[] imageBytes)
    {
        lock (lockObject)
        {
            try
            {
                //my_text.text += "\n Sending!";
                byte[] lengthPrefix = Encoding.ASCII.GetBytes(imageBytes.Length.ToString("D16"));

                stream.Write(lengthPrefix, 0, lengthPrefix.Length);
                stream.Write(imageBytes, 0, imageBytes.Length);

                Debug.Log("Sent frame to server: " + imageBytes.Length + " bytes");
                //my_text.text += "\n Sent frame to server: " + imageBytes.Length + " bytes";
            }
            catch (Exception e)
            {
                Debug.LogError("Error sending frame: " + e.Message);
                //my_text.text += "\n Error sending frame: " + e.Message;
            }
        }
    }

    private void OnApplicationQuit()
    {
        if (photoCaptureObject != null)
        {
            photoCaptureObject.StopPhotoModeAsync(OnStoppedPhotoMode);
        }
        if (stream != null)
        {
            stream.Close();
        }
        if (client != null)
        {
            client.Close();
        }
    }

    void OnStoppedPhotoMode(PhotoCapture.PhotoCaptureResult result)
    {
        photoCaptureObject.Dispose();
        photoCaptureObject = null;
    }
}