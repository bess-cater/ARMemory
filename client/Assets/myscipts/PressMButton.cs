using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PressMButton : MonoBehaviour
{
    // Start is called before the first frame update
    
    public SocketClient socketClient;
    public void SetModalityToText()
    {
        socketClient.currentModality = SocketClient.ChosenModality.Text;
    }

    public void SetModalityToImage()
    {
        socketClient.currentModality = SocketClient.ChosenModality.Image;
    }

    public void SetModalityToTextAndImage()
    {
        socketClient.currentModality = SocketClient.ChosenModality.TextImage;
    }
}
