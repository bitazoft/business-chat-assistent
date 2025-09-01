export function validateFile(fileType, fileSize) {
    const allowedTypes = ["image/jpeg", "image/png", "image/webp"];
    const maxSize = 5 * 1024 * 1024; // 5 MB
  
    if (!allowedTypes.includes(fileType)) {
      throw new Error("Invalid file type");
    }
    if (fileSize > maxSize) {
      throw new Error("File too large (max 5MB)");
    }
  }
  