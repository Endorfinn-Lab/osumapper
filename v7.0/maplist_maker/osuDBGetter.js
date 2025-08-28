'use strict';

const fs = require("fs");
const path = require("path");

/**
 * Parses a .osu file and extracts beatmap information.
 * @param {string} filePath Full path to the .osu file.
 * @returns {object} A beatmap object.
 */
function parseOsuFile(filePath) {
    const content = fs.readFileSync(filePath, 'utf8');
    const lines = content.split(/\r?\n/);

    const beatmap = {
        artist_name: '',
        artist_name_unicode: '',
        song_title: '',
        song_title_unicode: '',
        creator_name: '',
        difficulty: '',
        osu_file_name: path.basename(filePath),
        ranked_status: 0,
        n_hitcircles: 0,
        n_sliders: 0,
        n_spinners: 0,
        last_modification_time: fs.statSync(filePath).mtime.getTime(),
        approach_rate: 5,
        circle_size: 5,
        hp_drain: 5,
        overall_difficulty: 5,
        slider_velocity: 1.0,
        star_rating_standard: {0: 0},
        star_rating_taiko: {0: 0},
        star_rating_ctb: {0: 0},
        star_rating_mania: {0: 0},
        drain_time: 0,
        total_time: 0,
        preview_offset: 0,
        timing_points: [],
        beatmap_id: 0,
        beatmapset_id: -1,
        mode: 0,
        song_source: '',
        song_tags: '',
        folder_name: path.basename(path.dirname(filePath)),
    };

    let section = '';
    let hitObjects = [];
    let timingPoints = [];

    for (const line of lines) {
        const trimmedLine = line.trim();
        if (!trimmedLine || trimmedLine.startsWith('//')) continue;

        if (trimmedLine.startsWith('[')) {
            section = trimmedLine.toLowerCase();
            continue;
        }

        switch (section) {
            case '[general]': {
                const [key, ...val] = trimmedLine.split(':');
                if(val.length > 0) {
                    const value = val.join(':').trim();
                    if (key.toLowerCase() === 'mode') beatmap.mode = parseInt(value, 10) || 0;
                    if (key.toLowerCase() === 'previewtime') beatmap.preview_offset = parseInt(value, 10) || 0;
                }
                break;
            }
            case '[metadata]': {
                const [key, ...val] = trimmedLine.split(':');
                if (val.length > 0) {
                    const value = val.join(':').trim();
                    const keyLower = key.toLowerCase();
                    if (keyLower === 'title') beatmap.song_title = value;
                    else if (keyLower === 'titleunicode') beatmap.song_title_unicode = value;
                    else if (keyLower === 'artist') beatmap.artist_name = value;
                    else if (keyLower === 'artistunicode') beatmap.artist_name_unicode = value;
                    else if (keyLower === 'creator') beatmap.creator_name = value;
                    else if (keyLower === 'version') beatmap.difficulty = value;
                    else if (keyLower === 'source') beatmap.song_source = value;
                    else if (keyLower === 'tags') beatmap.song_tags = value;
                    else if (keyLower === 'beatmapid') beatmap.beatmap_id = parseInt(value, 10) || 0;
                    else if (keyLower === 'beatmapsetid') beatmap.beatmapset_id = parseInt(value, 10) || -1;
                }
                break;
            }
            case '[difficulty]': {
                const [key, ...val] = trimmedLine.split(':');
                if(val.length > 0) {
                    const value = parseFloat(val.join(':').trim());
                    const keyLower = key.toLowerCase();
                    if (keyLower === 'hpdrainrate') beatmap.hp_drain = value;
                    else if (keyLower === 'circlesize') beatmap.circle_size = value;
                    else if (keyLower === 'overalldifficulty') beatmap.overall_difficulty = value;
                    else if (keyLower === 'approachrate') beatmap.approach_rate = value;
                    else if (keyLower === 'slidermultiplier') beatmap.slider_velocity = value;
                }
                break;
            }
            case '[timingpoints]': {
                const parts = trimmedLine.split(',');
                if (parts.length >= 2) {
                    const msPerBeat = parseFloat(parts[1]);
                    const offset = parseFloat(parts[0]);
                    const inherited = (parts.length >= 7) ? (parseInt(parts[6], 10) === 0) : true;
                    if (msPerBeat > 0) { // Only care about uninherited points for BPM
                        timingPoints.push([msPerBeat, offset, inherited]);
                    }
                }
                break;
            }
            case '[hitobjects]': {
                hitObjects.push(trimmedLine);
                break;
            }
        }
    }
    
    beatmap.timing_points = timingPoints;

    let lastObjectTime = 0;
    let firstObjectTime = hitObjects.length > 0 ? Infinity : 0;

    for (const obj of hitObjects) {
        const parts = obj.split(',');
        if (parts.length < 4) continue;
        
        const time = parseInt(parts[2], 10);
        if (time < firstObjectTime) firstObjectTime = time;
        
        const type = parseInt(parts[3], 10);
        if ((type & 1) > 0) beatmap.n_hitcircles++; // circle
        else if ((type & 2) > 0) beatmap.n_sliders++; // slider
        else if ((type & 8) > 0) { // spinner
            beatmap.n_spinners++;
            const endTime = parseInt(parts[5], 10);
            if(endTime > lastObjectTime) lastObjectTime = endTime;
        }

        if (time > lastObjectTime) lastObjectTime = time;
    }
    
    if (hitObjects.length > 0) {
        beatmap.total_time = lastObjectTime;
        beatmap.drain_time = Math.round((lastObjectTime - firstObjectTime) / 1000); // in seconds
    }

    if (!beatmap.artist_name_unicode) beatmap.artist_name_unicode = beatmap.artist_name;
    if (!beatmap.song_title_unicode) beatmap.song_title_unicode = beatmap.song_title;

    return beatmap;
}

/**
 * Scans the osu!/Songs directory and parses all .osu files.
 * @param {string} songsPath Path to the osu!/Songs directory.
 * @returns {{beatmaps: Array<object>}}
 */
function osuDBGetter(songsPath) {
    const beatmaps = [];
    console.log(`Scanning songs path: ${songsPath}`);
    try {
        const songFolders = fs.readdirSync(songsPath, { withFileTypes: true })
            .filter(dirent => dirent.isDirectory())
            .map(dirent => dirent.name);

        console.log(`Found ${songFolders.length} song folders.`);

        for (const folder of songFolders) {
            try {
                const folderPath = path.join(songsPath, folder);
                const files = fs.readdirSync(folderPath);
                const osuFiles = files.filter(f => f.toLowerCase().endsWith('.osu'));

                for (const osuFile of osuFiles) {
                    try {
                        const filePath = path.join(folderPath, osuFile);
                        const beatmapData = parseOsuFile(filePath);
                        beatmaps.push(beatmapData);
                    } catch (e) {
                        console.error(`Error parsing beatmap file: ${path.join(folder, osuFile)}`, e.message);
                    }
                }
            } catch (e) {
                console.error(`Error reading song folder: ${folder}`, e.message);
            }
        }
    } catch (e) {
        console.error(`Error reading songs directory: ${songsPath}`, e.message);
        return { beatmaps: [] };
    }
    
    console.log(`Successfully parsed ${beatmaps.length} beatmaps.`);
    return { beatmaps };
}

module.exports = osuDBGetter;
