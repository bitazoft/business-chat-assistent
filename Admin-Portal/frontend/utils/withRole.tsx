import { getCurrentUser } from "@/lib/auth";

export function withRole(Component: React.FC, allowedRoles: string[]) {
    return function RoleProtected(props: any) {
        const currentUser = getCurrentUser();
        
        if (!currentUser || !allowedRoles.includes(currentUser.role)) {
            return <p>Access denied</p>;
        }

        return <Component {...props} />;
    };
}
