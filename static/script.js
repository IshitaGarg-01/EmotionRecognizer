/* validation function checks whether the user has inputted both name and image.
If any of details is incomplete it will show an alert and ask user to fill all the details.
If filled successfully it will show an alert regarding success*/

function validation() {
  if (
    document.getElementById("name").value != "" &&
    document.getElementById("img").value.endsWith != ""
  ) {
    alert("Form filled successfully");
    document.getElementById("alert").submit();
  } else {
    alert("Fill up all the details");
    return false;
  }
}

/* This function checks whether the image entered by user if of valid type -of the format jpg,jpeg,png 
If not of valid type it will show error and asks user to enter valid file type */
function validateFileType() {
  var fileName = document.getElementById("img").value;
  var idxDot = fileName.lastIndexOf(".") + 1;
  var extFile = fileName.substr(idxDot, fileName.length).toLowerCase();
  if (extFile == "jpg" || extFile == "jpeg" || extFile == "png") {
    //TO DO
  } else {
    fileName = "";
    alert("Only jpg/jpeg and png files are allowed!. Please change the file");
  }
}
