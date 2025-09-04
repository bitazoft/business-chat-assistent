"use client"

import { useState, useEffect } from "react"
import { useRouter, useSearchParams } from "next/navigation"
import { AuthPage } from "@/components/auth-page"
import { setUserSession } from "@/lib/auth"

export function AuthPageWrapper() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const [mode, setMode] = useState<"login" | "register">("login")

  useEffect(() => {
    const modeParam = searchParams.get("mode")
    if (modeParam === "register" || modeParam === "login") {
      setMode(modeParam)
    }
  }, [searchParams])

  const handleLogin = (userData: { id: number; name: string; email: string; role: string; sellerId: number | null}) => {
    try {
      setUserSession(userData)

      // Redirect to dashboard
      router.push("/dashboard")
    } catch (error) {
      console.error("Error setting session:", error)
      // Fallback: still try to redirect
      router.push("/dashboard")
    }
  }

  const handleBack = () => {
    router.push("/")
  }

  const handleSwitchMode = (newMode: "login" | "register") => {
    setMode(newMode)
    router.push(`/auth?mode=${newMode}`)
  }

  return <AuthPage mode={mode} onLogin={handleLogin} onBack={handleBack} onSwitchMode={handleSwitchMode} />
}
