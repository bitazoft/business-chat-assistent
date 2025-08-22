import type React from "react"
import { DashboardLayoutClient } from "@/components/dashboard-layout-client"

export default function DashboardLayoutWrapper({
  children,
}: {
  children: React.ReactNode
}) {
  return <DashboardLayoutClient>{children}</DashboardLayoutClient>
}
