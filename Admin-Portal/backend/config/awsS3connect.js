import { S3Client } from '@aws-sdk/client-s3';

// Initialize an S3 client with provided credentials
export const s3Client = new S3Client({
    region: process.env.AWS_REGION,
    credentials: {
        accessKeyId: process.env.AWS_ACCESSKEYID,
        secretAccessKey: process.env.AWS_SECRETACCESSKEY
    }
});

export const awsFolderNames = {
    images: 'images'
};