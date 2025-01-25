document.addEventListener("DOMContentLoaded", () => {
  const isDarkMode = localStorage.getItem("dark-mode") === "true";
  if (isDarkMode) {
    document.body.classList.add("dark-mode");
  }

  const imageInput = document.getElementById("image");
  const imagePreview = document.getElementById("image-preview");
  const form = document.getElementById("upload-form");
  let cropper;

  imageInput.addEventListener("change", () => {
    const file = imageInput.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        imagePreview.src = e.target.result;
        imagePreview.style.display = "block";

        if (cropper) cropper.destroy();
        cropper = new Cropper(imagePreview, {
          aspectRatio: 1,
        });
      };
      reader.readAsDataURL(file);
    }
  });

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    if (cropper) {
      const croppedCanvas = cropper.getCroppedCanvas();
      const croppedImage = croppedCanvas.toDataURL("image/jpeg");
      const croppedImageInput = document.createElement("input");
      croppedImageInput.type = "hidden";
      croppedImageInput.name = "cropped_image";
      croppedImageInput.value = croppedImage;
      form.appendChild(croppedImageInput);
    }
    form.submit();
  });
});

function toggleMode() {
  const body = document.body;
  const isDarkMode = body.classList.toggle("dark-mode");
  localStorage.setItem("dark-mode", isDarkMode);
}
