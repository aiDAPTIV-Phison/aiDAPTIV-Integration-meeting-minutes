import React from "react";
import { invoke } from '@tauri-apps/api/core';
import Image from 'next/image';


export function About() {
    return (
        <div className="p-4 space-y-4 h-[80vh] overflow-y-auto">
            {/* Header */}
            <div className="text-center">
                <div className="mb-3">
                    <Image 
                        src="icon_128x128.png" 
                        alt="aiDAPTIV Meetily Logo" 
                        width={64} 
                        height={64}
                        className="mx-auto"
                    />
                </div>
                <h1 className="text-xl font-bold text-gray-900">aiDAPTIV Meetily</h1>
                <span className="text-sm text-gray-500"> v1.0.1 </span>
                <p className="text-medium text-gray-600 mt-1">
                    Real-time notes and summaries that never leave your machine.
                </p>
            </div>

            {/* Footer */}
            <div className="pt-2 border-t border-gray-200 text-center space-y-1">
                <p className="text-xs text-gray-500 font-medium">
                    Built by aiDAPTIV
                </p>
                <p className="text-xs text-gray-400">
                    Based on <a href="#" onClick={(e) => { e.preventDefault(); invoke('open_external_url', { url: 'https://github.com/Zackriya-Solutions/meeting-minutes' }); }} className="underline hover:text-gray-600">Meetily</a> by Zackriya Solutions (MIT License)
                </p>
            </div>
        </div>
    )
}