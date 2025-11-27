// Maximum search depth in plies (you can tune this; 64 is plenty for now)
constexpr int MAX_PLY = 64;

// Aspiration window around last score (e.g. ±25 cp)
const int ASP_WINDOW = 25;