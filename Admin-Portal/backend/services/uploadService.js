import { s3Client } from "../config/awsS3connect.js";
import { validateFile } from "../utils/validateFileType.js";
import { PutObjectCommand } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";

async function generatePresignedUrl(fileName, fileType, fileSize, folder) {
    try {
        validateFile(fileType, fileSize);

        const timestamp = Date.now();
        const uniqueFileName = `${folder}/${timestamp}-${fileName}`;

        const params = {
            Bucket: process.env.AWS_BUCKET_NAME,
            Key: uniqueFileName,
            ContentType: fileType,
        };

        const command = new PutObjectCommand(params);
        const uploadUrl = await getSignedUrl(s3Client, command, { expiresIn: 300 });

        return {
            uploadUrl,
            fileUrl: `https://${process.env.AWS_BUCKET_NAME}.s3.${process.env.AWS_REGION}.amazonaws.com/${uniqueFileName}`
        };
    } catch (error) {
        throw new Error(`Could not generate presigned URL: ${error.message}`);
    }
}

export default generatePresignedUrl;