let jsonData;
let currentVideoId;
let currentIndex = 0;
const contentDiv = document.getElementById('content');
const videoInfoDiv = document.getElementById('videoInfo');
const collectedData = {};
const popupModal = document.getElementById('popupModal');
const iconButton = document.getElementById('iconButton');
const closePopup = document.getElementById('closePopup');



// Function to fetch JSON data
function fetchJsonData() {
    return fetch('answers/all_answers2.json')
        .then(response => response.json())
        .then(data => {
            jsonData = data;
            const videoIds = Object.keys(jsonData);
            currentVideoId = videoIds[currentIndex];
            console.log(currentVideoId);
            displayQuestions(currentVideoId);
        })
        .catch(error => console.error('Error fetching JSON data:', error));
}

// Function to display questions for a given videoId
function displayQuestions(videoId) {
    contentDiv.innerHTML = '';
    const videoData = jsonData[videoId];
    console.log(videoData);


    const videoElement = document.createElement('video');
    videoElement.controls = true;
    videoElement.src = "videos/selected_vids/needed_vids/"+videoId+".mp4";  


    contentDiv.appendChild(videoElement);
    const modelNameMapping = {
        "gpt": "Model A",
        "llava": "Model B",
        "mgm": "Model C"
    };
    //? For each question 3 answers in block!
    Object.keys(videoData).forEach(questionId => {
        
        const questionData = videoData[questionId];


        const questionBlock = document.createElement('div');
        //questionBlock.className = 'question_block';

        const questionTitle = document.createElement('h2');
        questionTitle.innerText = questionData.question;
        questionBlock.appendChild(questionTitle);

        const modelsBlock = document.createElement('div');
        modelsBlock.className = "ModelsBlock";


        //questionBlock.appendChild(modelsBlock);

        ['gpt', 'llava', 'mgm'].forEach(model => {

            //? Parse.
            const modelData = questionData[model];

            //? In here - vertical.
            const modelElement = document.createElement('div');
            modelElement.className = 'model';

            //* Title of model.
            const modelTitle = document.createElement('h3');
            // modelTitle.innerText = model.replace('model', 'Model ');
            modelTitle.innerText = modelNameMapping[model];
            modelElement.appendChild(modelTitle);

            //* Image of model.
            const imgElement = document.createElement('img');
            imgElement.className = 'model-answer';
            imgElement.src = "frames/selected_frames/"+modelData.video_id+"/"+modelData.selected_frame;
            modelElement.appendChild(imgElement);
            //* <input type="range" min="1" max="7" value="1" class="slider" id="slider_lang_1">
            

            //* For pic eval.
            const sliderLine = document.createElement("div");
            sliderLine.className = "slider_line";
            modelElement.appendChild(sliderLine);

            const liltext = document.createElement("div");
            liltext.innerText = "Frame";
            liltext.className = "cat";
            sliderLine.appendChild(liltext);
            

            const lilSlider = document.createElement("input");
            lilSlider.type = "range";
            lilSlider.min = 1;
            lilSlider.max = 7;
            lilSlider.value = 1;
            lilSlider.className = "slider";
            lilSlider.id = `slider_${model}_${questionId}_image`;
            sliderLine.appendChild(lilSlider);


            //*<span id="value2">1</span>
            const lilSpan = document.createElement("span");
            lilSpan.innerText = "1";
            lilSpan.className = "slider_span";
            lilSpan.id = `value_${model}_${questionId}_image`;
            sliderLine.appendChild(lilSpan);

            //* For common eval.

            const sliderLine_common = document.createElement("div");
            sliderLine_common.className = "slider_line";
            modelElement.appendChild(sliderLine_common);

            const liltext_common = document.createElement("div");
            liltext_common.innerText = "Complex";
            liltext_common.className = "cat";
            sliderLine_common.appendChild(liltext_common);

            const lilSlider_common = document.createElement("input");
            lilSlider_common.type = "range";
            lilSlider_common.min = 1;
            lilSlider_common.max = 7;
            lilSlider_common.value = 1;
            lilSlider_common.className = "slider";
            lilSlider_common.id = `slider_${model}_${questionId}_common`;
            sliderLine_common.appendChild(lilSlider_common);

        
            //*<span id="value2">1</span>
            const lilSpan_common = document.createElement("span");
            lilSpan_common.innerText = "1";
            lilSpan_common.className = "slider_span";
            lilSpan_common.id = `value_${model}_${questionId}_common`;
            sliderLine_common.appendChild(lilSpan_common);


            //* For text eval.
            const sliderLine_text = document.createElement("div");
            sliderLine_text.className = "slider_line";
            modelElement.appendChild(sliderLine_text);

            const liltext_text = document.createElement("div");
            liltext_text.innerText = "Text";
            liltext_text.className = "cat";
            sliderLine_text.appendChild(liltext_text);

            const lilSlider_text = document.createElement("input");
            lilSlider_text.type = "range";
            lilSlider_text.min = 1;
            lilSlider_text.max = 7;
            lilSlider_text.value = 1;
            lilSlider_text.className = "slider";
            lilSlider_text.id = `slider_${model}_${questionId}_text`;
            sliderLine_text.appendChild(lilSlider_text);

            //*<span id="value2">1</span>
            const lilSpan_text = document.createElement("span");
            lilSpan_text.innerText = "1";
            lilSpan_text.className = "slider_span";
            lilSpan_text.id = `value_${model}_${questionId}_text`;
            sliderLine_text.appendChild(lilSpan_text);
            

            //* Text response of model.
            const answerElement = document.createElement('p');
            answerElement.className = 'model-answer';
            answerElement.innerText = modelData.answer;
            modelElement.appendChild(answerElement);

            modelsBlock.appendChild(modelElement)
        });

        contentDiv.appendChild(questionTitle);
        contentDiv.appendChild(modelsBlock);
    });
        const sliders = document.querySelectorAll('.slider');
        //console.log(sliders);
        const sliderValues = document.querySelectorAll('span[id^="value"]');
        //console.log(sliderValues);
        sliders.forEach((slider, index) => {
            slider.addEventListener('input', function() {
                sliderValues[index].textContent = slider.value;
            });
        });
}

function updateVideoInfo() {
    const videoIds = Object.keys(jsonData);
    videoInfoDiv.innerText = `Video ${currentIndex + 1} of ${videoIds.length}`;
}
function collectSliderValues() {
    const videoData = jsonData[currentVideoId];
    const videoValues = {};

    Object.keys(videoData).forEach(questionId => {
        const questionValues = {};

        ['gpt', 'llava', 'mgm'].forEach(model => {
            const frameSlider = document.getElementById(`slider_${model}_${questionId}_image`);
            const complexSlider = document.getElementById(`slider_${model}_${questionId}_common`);
            const textSlider = document.getElementById(`slider_${model}_${questionId}_text`);

            questionValues[model] = {
                frame: frameSlider ? frameSlider.value : null,
                complex: complexSlider ? complexSlider.value : null,
                text: textSlider ? textSlider.value : null
            };
        });

        videoValues[questionId] = questionValues;
    });

    collectedData[currentVideoId] = videoValues;
}


function downloadJson(data, filename = 'data.json') {
    const userId = localStorage.getItem('userId');
    const jsonString = JSON.stringify(data, null, 2);
    const blob = new Blob([jsonString], { type: 'application/json' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = userId+"_data.json";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Event listener for the Next button
document.getElementById('nextButton').addEventListener('click', () => {
    collectSliderValues();
    currentIndex++;
    const videoIds = Object.keys(jsonData);
    // if (currentIndex >= videoIds.length) {
    //     currentIndex = 0;
    // }
    if (currentIndex >= videoIds.length) {
        const nextButton = document.getElementById('nextButton');
        nextButton.style.display = 'none';
        alert("You have reached the last video. The collected data will now be downloaded.");
        downloadJson(collectedData);
        currentIndex = 0; // Reset to the first video if you want to loop back
         // Hide the next button
        return;
    }
    currentVideoId = videoIds[currentIndex];
    updateVideoInfo();
    displayQuestions(currentVideoId);
    window.scrollTo(0, 0);
});
iconButton.addEventListener('click', () => {
    popupModal.style.display = 'block';
});

// Close the modal when the close button is clicked
closePopup.addEventListener('click', () => {
    popupModal.style.display = 'none';
});

// Close the modal when the user clicks anywhere outside of the modal content
window.addEventListener('click', (event) => {
    if (event.target == popupModal) {
        popupModal.style.display = 'none';
    }
});

// Fetch the JSON data when the page loads

fetchJsonData();
