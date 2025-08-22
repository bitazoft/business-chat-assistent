import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const { email, password } = await request.json()

    // Mock authentication logic
    if (!email || !password) {
      return NextResponse.json({ success: false, message: "Email and password are required" }, { status: 400 })
    }

    if (password.length < 6) {
      return NextResponse.json({ success: false, message: "Password must be at least 6 characters" }, { status: 400 })
    }

    // Mock successful login
    const user = {
      email,
      businessName: "Demo Business",
      whatsappNumber: "+1 (555) 123-4567",
      address: "123 Business St",
    }

    return NextResponse.json({
      success: true,
      user,
      message: "Login successful",
    })
  } catch (error) {
    return NextResponse.json({ success: false, message: "Internal server error" }, { status: 500 })
  }
}
