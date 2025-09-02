"use client"

import type React from "react"
import { useState, useEffect } from "react"
import { useRouter, usePathname } from "next/navigation"
import { Sidebar } from "@/components/sidebar"
import { TopBar } from "@/components/top-bar"
import { clearUserSession, getCurrentUser, type User } from "@/lib/auth"

interface DashboardLayoutClientProps {
  children: React.ReactNode
}

export function DashboardLayoutClient({ children }: DashboardLayoutClientProps) {
  const router = useRouter()
  const pathname = usePathname()
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(true)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)

  useEffect(() => {
    const checkAuth = () => {
      const currentUser = getCurrentUser()

      if (!currentUser) {
        router.push("/auth")
        return
      }

      setUser(currentUser)
      setLoading(false)
    }

    const timer = setTimeout(checkAuth, 100)
    return () => clearTimeout(timer)
  }, [router])

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

  const getActiveSection = () => {
    if (pathname === "/dashboard") return "dashboard"
    if (pathname.startsWith("/dashboard/products")) return "products"
    if (pathname.startsWith("/dashboard/orders")) return "orders"
    if (pathname.startsWith("/dashboard/analytics")) return "analytics"
    if (pathname.startsWith("/dashboard/settings")) return "settings"
    return "dashboard"
  }

  const handleSectionChange = (section: string) => {
    if (section === "dashboard") {
      router.push("/dashboard")
    } else {
      router.push(`/dashboard/${section}`)
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-[#0f0f23] flex items-center justify-center">
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 bg-violet-400 rounded-full animate-pulse"></div>
          <div className="text-violet-400 text-xl">Loading...</div>
        </div>
      </div>
    )
  }

  if (!user) {
    return (
      <div className="min-h-screen bg-[#0f0f23] flex items-center justify-center">
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 bg-red-400 rounded-full animate-pulse"></div>
          <div className="text-red-400 text-xl">Redirecting to login...</div>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-screen bg-[#0f0f23] overflow-hidden">
      <Sidebar
        activeSection={getActiveSection()}
        onSectionChange={handleSectionChange}
        collapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
      />
      <div className="flex-1 flex flex-col min-w-0">
        <TopBar user={user} onLogout={handleLogout} onNavigateToSettings={() => handleSectionChange("settings")} />
        <main className="flex-1 overflow-auto p-6">
          <div className="transition-all duration-500 ease-in-out transform">{children}</div>
        </main>
      </div>
    </div>
  )
}
