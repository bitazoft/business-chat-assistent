import userService from '../services/userService.js';

const getAllUsers = async (req, res) => {
    try {
        const users = await userService.usersWithStats();

        res.status(200).json({ users });
    } catch (error) {
        res.status(500).json({ error: error.message || 'Internal Server Error' });
    }
}

export { getAllUsers };