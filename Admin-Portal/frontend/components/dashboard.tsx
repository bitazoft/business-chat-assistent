"use client"

import { DashboardOverview } from "@/components/dashboard-overview"

interface DashboardProps {
  user: { id: number; name: string; email: string; role: string}
  onLogout: () => void
}

export function Dashboard({ user, onLogout }: DashboardProps) {
  return <DashboardOverview />
}
