import generatePresignedUrl from '../services/uploadService.js';

function uploadFile(req, res) {
    const { fileName, fileType, fileSize, folder } = req.body; 

    if (!fileName || !fileType || !fileSize || !folder) {
        return res.status(400).json({ error: 'Missing required fields' });
    }

    generatePresignedUrl(fileName, fileType, fileSize, folder)
        .then(({ uploadUrl, fileUrl }) => {
            res.status(200).json({ uploadUrl, fileUrl });
        })
        .catch((error) => {
            res.status(500).json({ error: error.message });
        });
}

export default uploadFile;