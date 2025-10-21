export const CLASS_NAME_MAP: Record<string, string> = {
"0": "HSIL",
"1": "LSIL",
"2": "NSIL",
HSIL: "HSIL",
LSIL: "LSIL",
NSIL: "NSIL",
};


export const CLASS_COLORS: Record<string, string> = {
HSIL: "bg-red-100 text-red-800 border-red-200",
LSIL: "bg-yellow-100 text-yellow-800 border-yellow-200",
NSIL: "bg-green-100 text-green-800 border-green-200",
};


export const mapClass = (v: unknown) => CLASS_NAME_MAP[String(v)] ?? String(v);


export const getClassColor = (className: string) =>
CLASS_COLORS[className] || "bg-gray-100 text-gray-800 border-gray-200";