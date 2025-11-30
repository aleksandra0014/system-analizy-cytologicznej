import { Button } from "@/components/ui/button";
import { User as UserIcon, Home as HomeIcon, LogOut } from "lucide-react";
import type { User } from "@/types";

export default function TopBar({ user, goHome, doLogout }: { user: User | null; goHome: () => void; doLogout: () => void }) {
  return (
    <div className="w-full sticky top-0 z-40 backdrop-blur-lg bg-white/80 border-b border-blue-100 shadow-sm">
      <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
        <Button onClick={goHome} variant="ghost" size="sm" className="flex items-center gap-2 hover:bg-blue-50">
          <HomeIcon className="w-4 h-4" />
          Home
        </Button>
        <div className="text-center">
          <h1 className="text-2xl md:text-3xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
            LBC Slides Analysis
          </h1>
          <p className="text-sm text-gray-500 hidden md:block">Advanced Cervical Cell Classification</p>
        </div>
        <div className="flex items-center gap-1">
          {user ? (
            <>
              <div className="hidden md:flex items-center gap-2 px-3 py-2 bg-blue-50 rounded-full">
                <UserIcon className="w-4 h-4 text-blue-600" />
                <span className="text-sm text-gray-700">{user.name} {user.surname}</span>
              </div>
              <Button onClick={doLogout} variant="ghost" size="sm" className="flex items-center gap-2 hover:bg-red-50 hover:text-red-600">
                <LogOut className="w-4 h-4" />
                Logout
              </Button>
            </>
          ) : (
            <div className="w-20" />
          )}
        </div>
      </div>
    </div>
  );
}