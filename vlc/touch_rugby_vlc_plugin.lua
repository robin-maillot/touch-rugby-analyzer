-- VLC Extension
local d = nil
local current_owner = ""

function descriptor()
    return {
        title = "1 - Rugby Logger Pro",
        version = "1.15",
        author = "Gemini",
        capabilities = {"menu"}
    }
end

function activate()
    -- SMART PAUSE LOGIC
    local input = vlc.object.input()
    if input then
        local state = vlc.var.get(input, "state")
        -- Only pause if state is 3 (Playing). If 4 (Paused), do nothing.
        if state == 2 then
            vlc.playlist.pause()
        end
    end

    show_owner_menu()
end

function deactivate()
    if d then d:delete() end
end

function close()
    vlc.deactivate()
end

-- --- DATA ---
local MENU_STRUCTURE = {
    ["Penalty"] = {"Offside", "Touch and Pass", "Forward Pass", "Delay of Play", "In the Ruck", "Shoulder", "Off the Mark", "Not Moving Forward"},
    ["Turnover"] = {"Ball Down", "6th Touch", "Dummy Touch", "Bad Roll", "6 Again", "Interception"},
    ["Game Event"] = {"Game Start", "Game End"},
    ["Try"] = {"Scoop", "Other", "32 - Long", "33 Quicky", "33", "32 Cut", "French Flair"},
    ["To Review"] = {}
}

-- --- HELPERS ---

function get_csv_path()
    local path = ""
    if os and os.getenv then
        path = os.getenv("USERPROFILE") .. "\\Desktop\\rugby_events.csv"
    else
        path = vlc.config.userdatadir() .. "\\rugby_events.csv"
    end
    return path
end

function format_time(seconds)
    local h = math.floor(seconds / 3600)
    local m = math.floor((seconds % 3600) / 60)
    local s = math.floor(seconds % 60)
    return string.format("%02d:%02d:%02d", h, m, s)
end

-- --- UI ---

function show_owner_menu()
    if d then d:delete() end
    d = vlc.dialog("Possession Owner")
    if not d then return end

    d:add_label("<b>Who has possession?</b>", 1, 1, 2, 1)

    d:add_button("Team 1", function()
        current_owner = "Team 1"
        show_main_menu()
    end, 1, 2, 1, 1)

    d:add_button("Team 2", function()
        current_owner = "Team 2"
        show_main_menu()
    end, 2, 2, 1, 1)

    d:add_button("Undo Last", undo_last, 1, 3, 1, 1)
    d:add_button("EXIT", function() vlc.deactivate() end, 2, 3, 1, 1)
end

function show_main_menu()
    if d then d:delete() end
    d = vlc.dialog("Event Category (" .. current_owner .. ")")
    if not d then return end

    local categories = {"Penalty", "Turnover", "Try", "Game Event", "To Review"}
    for i, cat in ipairs(categories) do
        d:add_button(cat, function()
            if MENU_STRUCTURE[cat] and #MENU_STRUCTURE[cat] > 0 then
                show_submenu(cat)
            else
                log_choice(cat, "")
            end
        end, 1, i+1, 2, 1)
    end
    d:add_button("← Back", show_owner_menu, 1, 8, 1, 1)
    d:add_button("EXIT", function() vlc.deactivate() end, 2, 8, 1, 1)
end

function show_submenu(category)
    if d then d:delete() end
    d = vlc.dialog(category)
    local options = MENU_STRUCTURE[category]
    local row, col = 2, 1
    for _, opt in ipairs(options) do
        d:add_button(opt, function() log_choice(category, opt) end, col, row, 1, 1)
        if col == 1 then col = 2 else col = 1; row = row + 1 end
    end
    d:add_button("<- Back", show_main_menu, 1, row + 1, 1, 1)
    d:add_button("Exit", function() vlc.deactivate() end, 2, row + 1, 1, 1)
end

-- --- LOGGING LOGIC ---

function log_choice(category, sub_option)
    local input = vlc.object.input()
    local item = vlc.input.item()
    if not input or not item then
        vlc.osd.message("Error: No video playing")
        vlc.deactivate()
        return
    end

    local video_name = item:name() or "Unknown"
    local time_micro = vlc.var.get(input, "time")
    local time_string = format_time(time_micro / 1000000)
    local file_path = get_csv_path()

    local file = io.open(file_path, "a")
    if not file then
        vlc.osd.message("Error: CSV is open in Excel")
        vlc.deactivate()
        return
    end

    if file:seek("end") == 0 then
        file:write("Time,Possession Owner,Type,Name,Video Name,To Review,Comment,Youtube Link\n")
    end

    local review_status = (category == "To Review") and "YES" or ""
    local row = string.format("%s,\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",, \n",
                                time_string, current_owner, category, sub_option, video_name, review_status)

    file:write(row)
    file:close()
    vlc.osd.message("Logged [" .. current_owner .. "]: " .. category)

    vlc.deactivate()
end

function undo_last()
    local file_path = get_csv_path()
    local f = io.open(file_path, "r")
    if not f then vlc.deactivate() return end

    local lines = {}
    for line in f:lines() do table.insert(lines, line) end
    f:close()

    if #lines > 1 then
        table.remove(lines)
        local out = io.open(file_path, "w")
        if out then
            for _, l in ipairs(lines) do out:write(l .. "\n") end
            out:close()
            vlc.osd.message("Undo Successful")
        end
    end
    vlc.deactivate()
end