"use client"

import { useState } from "react"
import { Sidebar } from "@/components/sidebar"
import { TopBar } from "@/components/top-bar"
import { DashboardOverview } from "@/components/dashboard-overview"
import { ProductManagement } from "@/components/product-management"
import { OrdersPage } from "@/components/orders-page"
import { AnalyticsPage } from "@/components/analytics-page"
import { SettingsPage } from "@/components/settings-page"

interface DashboardProps {
  user: { businessName: string; email: string }
  onLogout: () => void
}

export function Dashboard({ user, onLogout }: DashboardProps) {
  const [activeSection, setActiveSection] = useState("dashboard")
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)

  const renderContent = () => {
    switch (activeSection) {
      case "dashboard":
        return <DashboardOverview />
      case "products":
        return <ProductManagement />
      case "orders":
        return <OrdersPage />
      case "analytics":
        return <AnalyticsPage />
      case "settings":
        return <SettingsPage />
      default:
        return <DashboardOverview />
    }
  }

  const handleLogout = async () => {
    try {
      await fetch("http://localhost:7001/api/auth/logout", {
        method: "POST",
        credentials: "include",
      });
      onLogout();
    } catch (error) {
      console.error("Logout failed", error);
    }
  };

  return (
    <div className="flex h-screen bg-[#0f0f23]">
      <Sidebar
        activeSection={activeSection}
        onSectionChange={setActiveSection}
        collapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
      />
      <div className="flex-1 flex flex-col transition-all duration-300">
        <TopBar user={user} onLogout={handleLogout} />
        <main className="flex-1 overflow-auto p-6">
          <div className="transition-all duration-500 ease-in-out transform">{renderContent()}</div>
        </main>
      </div>
    </div>
  )
}
