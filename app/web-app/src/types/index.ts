export type Results = {
slide_uid?: string;
pacjent_uid?: string;
bbox_public_url?: string;
crop_public_urls?: Record<string, string>;
crop_gridfs_names?: Record<string, string>;
predict_fused?: Record<string, string | number>;
probs?: Record<string, { fused?: Record<string, number> }>;
features_list?: Record<string, Record<string, number>>;
slide_summary?: { overall_class?: string | number; explanation?: string; confidence?: number };
slide_summary_text?: string;
overall_class?: string | number;
add_info?: string | null;
cells_explanations?: Record<string, string | { explanation?: string }>;
probability?: Record<string, number>;
};


export type GradcamResp = {
overlay_url: string;
heatmap_url: string;
activation_url: string;
predicted_class: string;
};


export type Patient = { pacjent_uid: string; created_at?: string | null };
export type SlideItem = {
slajd_uid: string;
status?: string | null;
overall_class?: string | number | null;
created_at?: string | null;
add_info?: string | null;
};


export type User = { email: string; imie?: string; nazwisko?: string; rola?: string };
export type Mode = "home" | "add" | "old" | "login";