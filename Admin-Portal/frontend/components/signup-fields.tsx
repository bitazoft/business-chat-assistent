import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

interface SignupFieldsProps {
  formData: {
    shop_name: string
    email: string
    phone: string
    address: string
    name: string
    whatsapp_number_id: string
    password: string
    confirmPassword: string
  }
  onChange: (field: string, value: string) => void
  passwordsMatch: boolean
  showPasswordError: boolean
}

export function SignupFields({ formData, onChange, passwordsMatch, showPasswordError }: SignupFieldsProps) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
      <div className="space-y-3">
        <div className="space-y-1">
          <Label htmlFor="shop_name" className="text-violet-400 font-medium text-sm">
            Business Name
          </Label>
          <Input
            id="shop_name"
            type="text"
            value={formData.shop_name}
            onChange={(e) => onChange("shop_name", e.target.value)}
            className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 transition-all duration-300 h-10"
            placeholder="Enter your business name"
            required
          />
        </div>

        <div className="space-y-1">
          <Label htmlFor="email" className="text-violet-400 font-medium text-sm">
            Email Address
          </Label>
          <Input
            id="email"
            type="email"
            value={formData.email}
            onChange={(e) => onChange("email", e.target.value)}
            className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 transition-all duration-300 h-10"
            placeholder="Enter your email"
            required
          />
        </div>

        <div className="space-y-1">
          <Label htmlFor="name" className="text-violet-400 font-medium text-sm">
            Name Of Company Owner
          </Label>
          <Input
            id="name"
            type="text"
            value={formData.name}
            onChange={(e) => onChange("name", e.target.value)}
            className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 transition-all duration-300 h-10"
            placeholder="Enter name of company owner"
            required
          />
        </div>

        <div className="space-y-1">
          <Label htmlFor="password" className="text-violet-400 font-medium text-sm">
            Password
          </Label>
          <Input
            id="password"
            type="password"
            value={formData.password}
            onChange={(e) => onChange("password", e.target.value)}
            className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 transition-all duration-300 h-10"
            placeholder="Enter your password (6+ characters)"
            required
          />
        </div>
      </div>

      <div className="space-y-3">
        <div className="space-y-1">
          <Label htmlFor="whatsappNumber" className="text-violet-400 font-medium text-sm">
            WhatsApp Number
          </Label>
          <Input
            id="phone"
            type="tel"
            value={formData.phone}
            onChange={(e) => onChange("phone", e.target.value)}
            className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 transition-all duration-300 h-10"
            placeholder="+1 (555) 123-4567"
            required
          />
        </div>

        <div className="space-y-1">
          <Label htmlFor="address" className="text-violet-400 font-medium text-sm">
            Business Address
          </Label>
          <Input
            id="address"
            type="text"
            value={formData.address}
            onChange={(e) => onChange("address", e.target.value)}
            className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 transition-all duration-300 h-10"
            placeholder="Enter your business address"
            required
          />
        </div>

        <div className="space-y-1">
          <Label htmlFor="whatsapp_number_id" className="text-violet-400 font-medium text-sm">
            Whatsapp Number ID
          </Label>
          <Input
            id="whatsapp_number_id"
            type="text"
            value={formData.whatsapp_number_id}
            onChange={(e) => onChange("whatsapp_number_id", e.target.value)}
            className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 transition-all duration-300 h-10"
            placeholder="Enter your WhatsApp number ID"
            required
          />
        </div>

        <div className="space-y-1">
          <Label htmlFor="confirmPassword" className="text-violet-400 font-medium text-sm">
            Confirm Password
          </Label>
          <Input
            id="confirmPassword"
            type="password"
            value={formData.confirmPassword}
            onChange={(e) => onChange("confirmPassword", e.target.value)}
            className={`bg-[#0f0f23] border-gray-600 text-white focus:ring-violet-400/20 transition-all duration-300 h-10 ${
              showPasswordError
                ? "border-red-500 focus:border-red-500"
                : passwordsMatch && formData.confirmPassword.length > 0
                  ? "border-emerald-500 focus:border-emerald-500"
                  : "focus:border-violet-400"
            }`}
            placeholder="Confirm your password"
            required
          />
          {formData.confirmPassword.length > 0 && (
            <div className="flex items-center space-x-2 text-xs">
              {passwordsMatch ? (
                <>
                  <div className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse"></div>
                  <span className="text-emerald-400 font-medium">Passwords match</span>
                </>
              ) : (
                <>
                  <div className="w-1.5 h-1.5 bg-red-400 rounded-full animate-pulse"></div>
                  <span className="text-red-400 font-medium">Passwords do not match</span>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}


