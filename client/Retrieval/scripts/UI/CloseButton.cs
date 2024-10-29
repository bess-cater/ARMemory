using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CloseButton : MonoBehaviour
{
    // Start is called before the first frame update
    public GameObject Menu;
    public GameObject memo;
    public GameObject text_only;
    public GameObject close;
    private bool isSelected = false;
    void Start()
    {
        
    }

    public bool GetIsSelected()
    {
        return isSelected;
    }

    // Update is called once per frame
    public void ToggleSelected()
    {
                    //button.image.sprite = Activated;
        isSelected = true;
            
        Menu.SetActive(false);
        memo.SetActive(false);
        text_only.SetActive(false);
        close.SetActive(false);
    }
}
