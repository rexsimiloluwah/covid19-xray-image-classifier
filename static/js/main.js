// FOR THE UPLOAD BUTTON

var fileLabel = document.getElementById("fileLabel");
var fileInput = document.getElementById("file");
var fileLabelInner = fileLabel.innerHTML

// Preview Image

const imgContainer = document.getElementById('previewImg')
const img = imgContainer.querySelector('.xrayvisual')
const submitBtn = document.querySelector('.submitBtn')

console.log(fileLabelInner);

console.log("shit")
fileInput.addEventListener('change', function(e){
    var fileName = " ";
    fileName = e.target.value.split("fakepath").pop();
            
    fileLabel.innerHTML = `<span style="font-size:14px;"> ${fileName} <span class='fa fa-check' style='color: white;margin-left: 12px; margin-top: 10px;margin-bottom:auto;'></span></span>`;

    fileLabel.style.border = "2px solid rgb(255,255,255)"

    let file = this.files[0]
    if (file){
        const reader = new FileReader();
        img.style.display = 'block';
        submitBtn.style.display = 'block';
        reader.readAsDataURL(file);
        reader.addEventListener("load", function(){

            img.setAttribute('src', this.result);
        });
    }

    else{
    img.setAttribute("src", "");
    }
    
})



function filesize(element){
    console.log(element.files[0].size)

    //Saving it as a cookie for accessibility in the Flask application
    document.cookie = `filesize = ${element.files[0].size}`  //Using the string interpolation syntax in Javascript 
    }



   




