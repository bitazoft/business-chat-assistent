"use client"

import type React from "react"

import { useRouter, usePathname } from "next/navigation"
import { Sidebar } from "@/components/sidebar"
import { TopBar } from "@/components/top-bar"
import { clearUserSession, type User } from "@/lib/auth"

interface DashboardLayoutProps {
  user: User
  children: React.ReactNode
}

export function DashboardLayout({ user, children }: DashboardLayoutProps) {
  const router = useRouter()
  const pathname = usePathname()

  const handleLogout = () => {
    clearUserSession()
    router.push("/")
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

  return (
    <div className="flex h-screen bg-[#0f0f23]">
      <Sidebar
        activeSection={getActiveSection()}
        onSectionChange={handleSectionChange}
        collapsed={false}
        onToggleCollapse={() => {}}
      />
      <div className="flex-1 flex flex-col transition-all duration-300">
        <TopBar user={user} onLogout={handleLogout} onNavigateToSettings={() => handleSectionChange("settings")} />
        <main className="flex-1 overflow-auto p-6">
          <div className="transition-all duration-500 ease-in-out transform">{children}</div>
        </main>
      </div>
    </div>
  )
}
