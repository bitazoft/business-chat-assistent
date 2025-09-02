"use client"

import { useRouter } from "next/navigation"
import { Dashboard } from "@/components/dashboard"
import { clearUserSession, type User } from "@/lib/auth"

interface DashboardWrapperProps {
  user: User
}

export function DashboardWrapper({ user }: DashboardWrapperProps) {
  const router = useRouter()

  const handleLogout = async () => {
      try {
        await fetch(`${process.env.NEXT_PUBLIC_API_BASE_URL}/auth/logout`, {
          method: "POST",
          credentials: "include",
        });
        router.push("/")
      } catch (error) {
        console.error("Logout failed", error);
      }
  }

  return <Dashboard user={user} onLogout={handleLogout} />
}
