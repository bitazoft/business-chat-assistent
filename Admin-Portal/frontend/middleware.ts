import { NextResponse } from "next/server"
import type { NextRequest } from "next/server"

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl

  // Fetch cookies
  const tokenCookie = request.cookies.get("access_token")

  const isDashboardRoute = pathname.startsWith("/dashboard")
  const isAuthRoute = pathname === "/auth"

  const isSessionValid =
    tokenCookie?.value

  // 🚫 If user is not logged in and trying to access protected dashboard
  if (isDashboardRoute && !isSessionValid) {
    const response = NextResponse.redirect(new URL("/auth", request.url))
    response.cookies.delete("access_token")
    return response
  }

  // 🔐 If user is already logged in and tries to access /auth again
  if (isAuthRoute && isSessionValid) {
    return NextResponse.redirect(new URL("/dashboard", request.url))
  }

  return NextResponse.next()
}

export const config = {
  matcher: ["/dashboard/:path*", "/auth"],
}
