using System;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

//? This script enables Hololens 2 to:
//? 1) Record voice input and transcribe it to text
//? 2) Send the text to the server
//? 3) Receive image and text and display them ion the user's view
public class SocketClient : MonoBehaviour
{   private TcpClient client;
    private NetworkStream stream;
    private Thread clientThread;
    public String mine = "HEHE";
    private bool isConnected = false;
    private Texture2D receivedTexture = null;
    private Renderer quadRenderer;
    private int dataLength;
    private GameObject quad;
    private GameObject text_only;
    private GameObject memo;

    private GameObject close;
    public TMP_Text ReceivedText;
    public TMP_Text ReceivedText_Only;

    string int_text;
    public enum ChosenModality
        {
            Image,
            Text,
            TextImage
        };
        public ChosenModality currentModality;

        void Start()
    {
        
        ConnectToServer("0.0.0.0", 9999); //! Replace with your server IP address here
        receivedTexture = new Texture2D(2, 2);
        // clientThread.IsBackground = true;
        // clientThread.Start();
        quad = GameObject.Find("Photo_frame");
        quad.SetActive(false);
        memo = GameObject.Find("textual");
        memo.SetActive(false);
        close = GameObject.Find("Close");
        close.SetActive(false);
        // text_only = GameObject.Find("text_only");
        // text_only.SetActive(false);
        print(quad.gameObject.name);

    }

    private void ConnectToServer(string serverIP, int port)
    {
        try
        {
            client = new TcpClient(serverIP, port);
            stream = client.GetStream();
            Debug.Log("Connected to server");

        }
        catch (SocketException e)
        {
            Debug.Log("SocketException: " + e);
        }
        
    }
     public void SendMessageToServer(string message)
    {
        try
        {
            byte[] textBytes = Encoding.ASCII.GetBytes(message);
            byte[] lengthPrefix = Encoding.ASCII.GetBytes(textBytes.Length.ToString("D16"));

            stream.Write(lengthPrefix, 0, lengthPrefix.Length);
            stream.Write(textBytes, 0, textBytes.Length);

            Debug.Log("Sent text to server: " + message);

            // Receive the image from the server
            ReceiveImageFromServer();
        }
        catch (Exception e)
        {
            Debug.LogError("Error sending text: " + e.Message);
        }
    }
    private async void ReceiveImageFromServer()
{
    try
    {
        // Read the length prefix
        byte[] headerBuffer = new byte[32];
        await stream.ReadAsync(headerBuffer, 0, 32);
        string header = Encoding.UTF8.GetString(headerBuffer);

        int textLength = int.Parse(header.Substring(0, 16).Trim());
        int imageLength = int.Parse(header.Substring(16, 16).Trim());

        byte[] textBuffer = new byte[textLength];
        await stream.ReadAsync(textBuffer, 0, textLength);
        string receivedText = Encoding.UTF8.GetString(textBuffer);
        print("Received text: " + receivedText);
        ReceivedText.text = receivedText;
        ReceivedText_Only.text = receivedText;

        byte[] imageBuffer = new byte[imageLength];
        int totalRead = 0;
        while (totalRead < imageLength)
        {
            int bytesRead = await stream.ReadAsync(imageBuffer, totalRead, imageLength - totalRead);
            totalRead += bytesRead;
        }
        

        // Verify that the received image data is correct
        if (totalRead == imageLength)
        {
            receivedTexture = new Texture2D(2, 2); // Initialize with dummy values
            bool isLoaded = receivedTexture.LoadImage(imageBuffer);
            if (isLoaded)
            {
                if (currentModality == ChosenModality.Text)
                {
                    memo.SetActive(true);
                    close.SetActive(true);
                    print("Text modality active");

                }
                else if (currentModality == ChosenModality.Image)
                {
                    ApplyTextureToQuad(receivedTexture);
                }
                else if (currentModality == ChosenModality.TextImage)
                {
                    memo.SetActive(true);
                    ApplyTextureToQuad(receivedTexture);
                }
            }
            else
            {
                Debug.LogError("Failed to load image from received data");

            }
        }
        else
        {
            Debug.LogError("Mismatch in expected and received image data length");
           

        }
    }
    catch (Exception e)
    {
        Debug.LogError("Error: " + e.Message);

    }
}

private void ApplyTextureToQuad(Texture2D texture)
{
    try
    {
        quad.SetActive(true);
        Renderer quadRenderer = quad.GetComponent<Renderer>();

        // Try to use "Unlit/Texture" shader, fallback to "Standard" shader if not available
        Shader shader = Shader.Find("Unlit/Texture");
        if (shader == null)
        {
            shader = Shader.Find("Standard");
        }
        
        if (shader == null)
        {
            throw new Exception("No suitable shader found.");
        }

        quadRenderer.material = new Material(shader);
        quadRenderer.material.mainTexture = texture;
        close.SetActive(true);
        print("Image modality active");
       
    }
    catch (Exception e)
    {
        print(e.Message);
    }
}

     void OnApplicationQuit()
    {
        isConnected = false;
        if (clientThread != null)
            clientThread.Abort();
        if (stream != null)
            stream.Close();
        if (client != null)
            client.Close();
    }
}